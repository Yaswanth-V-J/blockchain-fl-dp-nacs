import os
import argparse
import numpy as np
import torch

from opacus.validators import ModuleValidator

from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.utils import compute_metrics, save_metrics_json
from run_phase1 import fed_avg, evaluate_global_model


def run_sanity_check(device):
    """
    Step 5: Quick sanity check.
    Train 1 round with 1 client, log update & reward to smart contract,
    and call getClientHistory() via web3.py to confirm read-back.
    """
    print("\n" + "=" * 75)
    print("RUNNING PHASE 3 SANITY CHECK: 1 Client, 1 Round, Web3 On-Chain Read/Write")
    print("=" * 75)

    logger = BlockchainLogger()

    # Load 1 client dataloader
    client_train_loaders, client_val_loaders, test_loader, _ = get_client_dataloaders(
        num_clients=1, alpha=0.5, batch_size=32
    )

    client = PrivateMedicalFLClient(
        client_id=0,
        train_loader=client_train_loaders[0],
        val_loader=client_val_loaders[0],
        device=device,
        target_epsilon=8.0,
        target_delta=1e-5,
        total_rounds=1,
        local_epochs=1,
        lr=0.005,
    )

    base = MedicalCNN(in_channels=1, num_classes=2)
    model = ModuleValidator.fix(base).to(device) if not ModuleValidator.is_valid(base) else base.to(device)
    init_weights = get_weights(model)

    # Loss before
    loss_before, _, _ = client.evaluate(init_weights, {})

    # Train 1 round
    config = {"local_epochs": 1, "lr": 0.005}
    updated_weights, _, fit_metrics = client.fit(init_weights, config)
    loss_after = fit_metrics["train_loss"]

    print(f"  [Sanity Client 0] Pre-training Loss: {loss_before:.4f} | Post-training Loss: {loss_after:.4f}")

    # Log update hash on-chain
    tx_hash, gas_used = logger.log_update(round_num=1, client_id=0, weights=updated_weights)
    print(f"  [Sanity Log Update] Tx Hash: {tx_hash} | Gas Used: {gas_used}")

    # Log placeholder reward on-chain
    reward_tx_hash, reward_gas = logger.record_placeholder_reward(
        round_num=1, client_id=0, local_loss_before=loss_before, local_loss_after=loss_after
    )
    print(f"  [Sanity Record Reward] Tx Hash: {reward_tx_hash} | Gas Used: {reward_gas}")

    # Read back from contract via getClientHistory()
    print("\n  --> Querying contract getClientHistory(clientId=0) via web3.py...")
    history = logger.get_client_history(client_id=0)

    print(f"  [On-Chain Read-Back Result]:")
    print(f"    Update Rounds : {history['update_rounds']}")
    print(f"    Update Hashes : {history['update_hashes']}")
    print(f"    Reward Rounds : {history['reward_rounds']}")
    print(f"    Reward Scores : {history['reward_scores']}")

    assert len(history["update_rounds"]) > 0, "Sanity check failed: no update rounds returned from contract!"
    assert len(history["reward_rounds"]) > 0, "Sanity check failed: no reward rounds returned from contract!"
    print("\n[SUCCESS] SANITY CHECK PASSED! On-chain read/write verified successfully.")
    print("=" * 75 + "\n")


def run_phase3_connected_experiment(target_epsilon=8.0, num_clients=5, num_rounds=10, alpha=0.5):
    print("=" * 75)
    print(f"PHASE 3: Blockchain-Connected DP FL Experiment (target_epsilon = {target_epsilon})")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds} | Smart Contract: ContributionLedger")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.005

    # Connect to Blockchain logger
    logger = BlockchainLogger()

    # Load dataset partitions
    client_train_loaders, client_val_loaders, test_loader, _ = get_client_dataloaders(
        num_clients=num_clients, alpha=alpha, batch_size=32
    )

    # Instantiate DP clients
    clients = [
        PrivateMedicalFLClient(
            client_id=i,
            train_loader=client_train_loaders[i],
            val_loader=client_val_loaders[i],
            device=device,
            target_epsilon=target_epsilon,
            target_delta=1e-5,
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
        )
        for i in range(num_clients)
    ]

    # Global model for server evaluation
    base = MedicalCNN(in_channels=1, num_classes=2)
    global_model = ModuleValidator.fix(base).to(device) if not ModuleValidator.is_valid(base) else base.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)
    print(f"Round 0 (Initial) -> Test Accuracy: {initial_metrics['accuracy']:.4f}, "
          f"F1: {initial_metrics['f1']:.4f}, Loss: {initial_metrics['loss']:.4f}\n")

    history = {
        "target_epsilon": target_epsilon,
        "rounds": [0],
        "accuracy": [initial_metrics["accuracy"]],
        "f1": [initial_metrics["f1"]],
        "precision": [initial_metrics["precision"]],
        "recall": [initial_metrics["recall"]],
        "loss": [initial_metrics["loss"]],
        "epsilon_spent": [0.0],
        "blockchain_log": [],
    }

    current_weights = get_weights(global_model)
    config = {"local_epochs": local_epochs, "lr": lr}

    for rnd in range(1, num_rounds + 1):
        print(f"--- DP FL Round {rnd}/{num_rounds} (Target epsilon={target_epsilon}) ---")
        client_updates = []
        client_sizes = []
        round_epsilons = []

        for client in clients:
            # Measure pre-training loss for reward calculation
            loss_before, _, _ = client.evaluate(current_weights, config)

            # Local DP fit
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            loss_after = fit_metrics["train_loss"]
            eps_spent = fit_metrics.get("epsilon_spent", 0.0)

            client_updates.append(updated_weights)
            client_sizes.append(num_samples)
            round_epsilons.append(eps_spent)

            # Log update hash to smart contract
            tx_hash, gas_used = logger.log_update(
                round_num=rnd, client_id=client.client_id, weights=updated_weights
            )

            # Record placeholder reward score on-chain
            # NOTE: Temporary score calculation - will be replaced by real NACS score in Phase 4
            reward_tx_hash, reward_gas = logger.record_placeholder_reward(
                round_num=rnd,
                client_id=client.client_id,
                local_loss_before=loss_before,
                local_loss_after=loss_after,
            )

            print(
                f"  Hospital Client {client.client_id + 1}: "
                f"loss {loss_before:.3f}->{loss_after:.3f} | "
                f"eps: {eps_spent:.4f} | "
                f"Tx update gas: {gas_used} | "
                f"Tx reward gas: {reward_gas}"
            )

            history["blockchain_log"].append({
                "round": rnd,
                "client_id": client.client_id,
                "update_tx_hash": tx_hash,
                "update_gas_used": gas_used,
                "reward_tx_hash": reward_tx_hash,
                "reward_gas_used": reward_gas,
            })

        # Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        # Global Evaluation
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)
        max_eps_this_round = float(np.max(round_epsilons))

        print(
            f"  >> Global Test Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Max cumulative epsilon: {max_eps_this_round:.4f}\n"
        )

        history["rounds"].append(rnd)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])
        history["epsilon_spent"].append(max_eps_this_round)

    # Save metrics JSON with blockchain_log
    json_path = "results/phase3_connected.json"
    save_metrics_json(history, json_path)
    print(f"Saved Phase 3 metrics and blockchain log to {json_path}")

    # Summary of on-chain activity
    total_txs = len(history["blockchain_log"]) * 2
    total_gas = sum(
        entry["update_gas_used"] + entry["reward_gas_used"]
        for entry in history["blockchain_log"]
    )
    print("\n" + "=" * 75)
    print("PHASE 3 CONNECTED EXPERIMENT COMPLETE!")
    print(f"  Final Accuracy   : {history['accuracy'][-1] * 100:.2f}%")
    print(f"  Final Macro F1   : {history['f1'][-1] * 100:.2f}%")
    print(f"  Total Epsilon    : {history['epsilon_spent'][-1]:.4f}")
    print(f"  Total On-Chain Tx: {total_txs} transactions")
    print(f"  Total Gas Used   : {total_gas:,} gas units")
    print("=" * 75)

    return history


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Blockchain-Connected FL Pipeline")
    parser.add_argument("--sanity-only", action="store_true", help="Run only the 1-round 1-client sanity check")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target privacy epsilon (default: 8.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 5: Always run quick sanity check first
    run_sanity_check(device)

    if not args.sanity_only:
        run_phase3_connected_experiment(target_epsilon=args.epsilon)


if __name__ == "__main__":
    main()
