import os
import json
import hashlib
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
from web3 import Web3


class BlockchainLogger:
    """
    Client for interacting with the ContributionLedger smart contract via web3.py.
    Logs model update SHA-256 hashes and contribution reward scores to the blockchain.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        abi_path: str = "blockchain/artifacts/contracts/ContributionLedger.sol/ContributionLedger.json",
        address_path: str = "blockchain/deployed_address.json",
    ):
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Check connection; raise clear error if Hardhat node is not running
        if not self.w3.is_connected():
            raise ConnectionError(
                f"\n[Blockchain Error] Could not connect to Hardhat Ethereum node at {rpc_url}.\n"
                "Please make sure the local Hardhat node is running by executing:\n"
                "  cd blockchain && npm run node\n"
                "in a separate terminal before running Phase 3 scripts."
            )

        # Load contract ABI
        if not os.path.exists(abi_path):
            raise FileNotFoundError(
                f"Contract ABI artifact not found at {abi_path}. "
                "Ensure Hardhat contracts are compiled (`npx hardhat compile` inside blockchain/)."
            )
        with open(abi_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
            self.abi = artifact["abi"]

        # Load deployed contract address
        if not os.path.exists(address_path):
            raise FileNotFoundError(
                f"Deployed address file not found at {address_path}. "
                "Ensure the contract is deployed (`npx hardhat run scripts/deploy.js --network localhost` inside blockchain/)."
            )
        with open(address_path, "r", encoding="utf-8") as f:
            addr_data = json.load(f)
            self.contract_address = Web3.to_checksum_address(addr_data["address"])

        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=self.contract_address, abi=self.abi
        )

        # Use Account #0 from default Hardhat accounts
        accounts = self.w3.eth.accounts
        if not accounts:
            raise RuntimeError("No Ethereum accounts found on the connected node.")
        self.account = accounts[0]
        print(f"[BlockchainLogger] Connected to node at {rpc_url}")
        print(f"[BlockchainLogger] Contract address: {self.contract_address}")
        print(f"[BlockchainLogger] Using Account #0: {self.account}")

    def serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        """Serialize numpy weight arrays into bytes for hashing."""
        return pickle.dumps(weights)

    def compute_update_hash(self, weights: List[np.ndarray]) -> bytes:
        """Compute SHA-256 hash of serialized model weights as 32 raw bytes (bytes32)."""
        serialized = self.serialize_weights(weights)
        return hashlib.sha256(serialized).digest()

    def log_update(
        self, round_num: int, client_id: int, weights: List[np.ndarray]
    ) -> Tuple[str, int]:
        """
        Serializes model weights, computes SHA-256 hash, and calls contract's logUpdate().
        Returns (transaction_hash_hex, gas_used).
        """
        update_bytes32 = self.compute_update_hash(weights)

        tx_hash = self.contract.functions.logUpdate(
            round_num, client_id, update_bytes32
        ).transact({"from": self.account})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hash_hex = receipt["transactionHash"].hex()
        gas_used = receipt["gasUsed"]
        return tx_hash_hex, gas_used

    def record_placeholder_reward(
        self,
        round_num: int,
        client_id: int,
        local_loss_before: float,
        local_loss_after: float,
    ) -> Tuple[str, int]:
        """
        Computes a simple placeholder reward score (0-100) based on local loss improvement
        and calls recordReward().

        NOTE: This simple score calculation is a TEMPORARY PLACEHOLDER for Phase 3.
        It will be replaced by the formal Noise-Aware Contribution Score (NACS) in Phase 4.
        """
        loss_improvement = local_loss_before - local_loss_after
        raw_score = int(round(loss_improvement * 100 + 50))
        score = int(np.clip(raw_score, 0, 100))

        tx_hash = self.contract.functions.recordReward(
            round_num, client_id, score
        ).transact({"from": self.account})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hash_hex = receipt["transactionHash"].hex()
        gas_used = receipt["gasUsed"]
        return tx_hash_hex, gas_used

    def get_client_history(self, client_id: int) -> Dict[str, Any]:
        """
        Calls getClientHistory() view function on contract to retrieve logged updates and rewards.
        """
        update_rounds, update_hashes, reward_rounds, reward_scores = (
            self.contract.functions.getClientHistory(client_id).call()
        )
        return {
            "client_id": client_id,
            "update_rounds": list(update_rounds),
            "update_hashes": [h.hex() for h in update_hashes],
            "reward_rounds": list(reward_rounds),
            "reward_scores": list(reward_scores),
        }
