import math
from typing import Dict, Any, Tuple


def check_training_health(
    history: Dict[str, Any],
    num_classes: int = 6,
    loss_tolerance: float = 0.1,
    acc_tolerance: float = 0.05,
    min_check_round: int = 2,
) -> Tuple[bool, str]:
    """
    Evaluates whether an FL training run is stalled or outputting uniform random guesses.

    Args:
        history: Metrics history dictionary containing 'rounds', 'accuracy', and 'loss' lists.
        num_classes: Number of target classification classes.
        loss_tolerance: Max distance from ln(num_classes) to count as 'chance loss'.
        acc_tolerance: Max distance from 1/num_classes to count as 'chance accuracy'.
        min_check_round: Minimum FL round before health checks are active (default: round 2+).

    Returns:
        (is_unhealthy, diagnostic_message):
            - is_unhealthy: True if the model is stuck at random chance level after round min_check_round.
            - diagnostic_message: Descriptive explanation of the health evaluation.
    """
    rounds = history.get("rounds", [])
    accuracies = history.get("accuracy", [])
    losses = history.get("loss", [])

    if not rounds or len(rounds) <= min_check_round:
        return False, "Training in initial warmup rounds."

    current_round = rounds[-1]
    current_acc = accuracies[-1]
    current_loss = losses[-1]

    chance_loss = math.log(num_classes)
    chance_acc = 1.0 / float(num_classes)

    is_chance_loss = abs(current_loss - chance_loss) <= loss_tolerance
    is_chance_acc = abs(current_acc - chance_acc) <= acc_tolerance

    if is_chance_loss and is_chance_acc and current_round >= min_check_round:
        diag = (
            f"[Training Health Alert] Model detected as NOT LEARNING at Round {current_round}! "
            f"Current Loss: {current_loss:.4f} (chance level ln({num_classes}) = {chance_loss:.4f} ± {loss_tolerance}), "
            f"Current Accuracy: {current_acc * 100:.2f}% (chance level {chance_acc * 100:.2f}% ± {acc_tolerance * 100:.1f}%)."
        )
        return True, diag

    return False, f"Round {current_round} Healthy: Loss = {current_loss:.4f}, Accuracy = {current_acc * 100:.2f}%"


def check_multilabel_training_health(
    history: Dict[str, Any],
    trivial_baseline: float = 0.0,
    f1_tolerance: float = 0.02,
    min_check_round: int = 2,
) -> Tuple[bool, str]:
    """
    Evaluates whether a multi-label FL training run is stalled at or near the trivial baseline floor.

    Args:
        history: Metrics history dictionary containing 'rounds', 'micro_f1', and 'loss' lists.
        trivial_baseline: Computed Micro-F1 floor obtained by predicting ALL-NEGATIVE for all findings.
        f1_tolerance: Max distance above trivial_baseline floor to still flag as non-learning.
        min_check_round: Minimum FL round before health checks are active (default: round 2+).

    Returns:
        (is_unhealthy, diagnostic_message):
            - is_unhealthy: True if the model Micro-F1 remains <= (trivial_baseline + f1_tolerance).
            - diagnostic_message: Descriptive explanation of the health evaluation.
    """
    rounds = history.get("rounds", [])
    micro_f1s = history.get("micro_f1", history.get("f1", []))

    if not rounds or len(rounds) <= min_check_round:
        return False, "Training in initial warmup rounds."

    current_round = rounds[-1]
    current_f1 = micro_f1s[-1] if micro_f1s else 0.0

    if len(micro_f1s) >= 2:
        last_two_stalled = (micro_f1s[-1] <= (trivial_baseline + f1_tolerance)) and (micro_f1s[-2] <= (trivial_baseline + f1_tolerance))
    else:
        last_two_stalled = (current_f1 <= (trivial_baseline + f1_tolerance))

    if last_two_stalled and current_round >= min_check_round:
        diag = (
            f"[Training Health Alert] Multi-Label Model detected as NOT LEARNING at Round {current_round}! "
            f"Current Micro-F1: {current_f1 * 100:.2f}% (Trivial baseline floor = {trivial_baseline * 100:.2f}% ± {f1_tolerance * 100:.1f}%)."
        )
        return True, diag

    return False, f"Round {current_round} Healthy: Micro-F1 = {current_f1 * 100:.2f}% (above baseline floor {trivial_baseline * 100:.2f}%)"

