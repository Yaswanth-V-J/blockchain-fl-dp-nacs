from typing import Dict, List, Optional
import numpy as np


def score_from_size(client_size: int, all_client_sizes: Optional[List[int]] = None) -> int:
    """
    Configuration A (Dataset-size baseline):
    The client's dataset size as a fraction of the total across all clients that round,
    scaled to 0-100.
    """
    if not all_client_sizes or sum(all_client_sizes) <= 0:
        return 0
    total_size = sum(all_client_sizes)
    fraction = float(client_size) / float(total_size)
    raw_score = int(round(fraction * 100.0))
    return int(np.clip(raw_score, 0, 100))


def score_from_quality(loss_before: float, loss_after: float) -> int:
    """
    Noise-blind quality baseline score (0-100) based on local loss improvement.
    Formula: loss_improvement * 100 + 50, clipped to [0, 100].
    """
    loss_improvement = loss_before - loss_after
    raw_score = int(round(loss_improvement * 100.0 + 50.0))
    return int(np.clip(raw_score, 0, 100))


def baseline_contribution_score(loss_before: float, loss_after: float) -> int:
    """
    Alias for score_from_quality() for backward compatibility with existing call sites.
    Note: "size_only" (score_from_size) is now the correct baseline for the size-based fairness claim in the paper.
    """
    return score_from_quality(loss_before, loss_after)


def compute_raw_score(
    client_size: int,
    all_client_sizes: Optional[List[int]],
    loss_before: float,
    loss_after: float,
    config: str = "full_nacs",
    noise_multiplier: float = 0.0,
) -> int:
    """
    Computes single-round contribution score under one of four explicit configurations:
    - "size_only"          -> Configuration A: score_from_size() alone
    - "size_quality"       -> Configuration B: mean of score_from_size() and score_from_quality()
    - "size_quality_noise" -> Configuration C: Config B scaled by trust_weight = 1/(1+noise_multiplier)
    - "full_nacs"          -> Configuration D: raw combined score (Config B) ready for temporal EMA
    """
    s_size = score_from_size(client_size, all_client_sizes)
    s_qual = score_from_quality(loss_before, loss_after)

    if config == "size_only":
        return s_size
    elif config == "size_quality":
        s_comb = (s_size + s_qual) / 2.0
        return int(np.clip(round(s_comb), 0, 100))
    elif config == "size_quality_noise":
        s_comb = (s_size + s_qual) / 2.0
        trust_weight = 1.0 / (1.0 + float(noise_multiplier))
        return int(np.clip(round(s_comb * trust_weight), 0, 100))
    elif config == "full_nacs":
        s_comb = (s_size + s_qual) / 2.0
        return int(np.clip(round(s_comb), 0, 100))
    else:
        raise ValueError(f"Unknown configuration: '{config}'. Expected one of ['size_only', 'size_quality', 'size_quality_noise', 'full_nacs'].")


class NACSTracker:
    """
    Noise-Aware Adaptive Contribution Scoring (NACS) Tracker.

    Maintains per-client score history across rounds for all four explicit configurations.
    Uses EMA temporal smoothing for the 'full_nacs' configuration.
    """

    def __init__(self):
        self.history: Dict[int, List[int]] = {}
        self.config_history: Dict[int, List[Dict[str, int]]] = {}

    def update(
        self,
        client_id: int,
        round_num: int,
        loss_before: float,
        loss_after: float,
        noise_multiplier: float,
        client_size: int = 0,
        all_client_sizes: Optional[List[int]] = None,
    ) -> int:
        """
        Calculates and records contribution scores across all 4 configurations for a client in a given round.

        Args:
            client_id: Unique identifier for the client.
            round_num: Current FL round index.
            loss_before: Local loss before round training.
            loss_after: Local loss after round training.
            noise_multiplier: DP noise multiplier applied during training.
            client_size: Number of samples in the client's training partition.
            all_client_sizes: List of sample sizes across all clients for the round.

        Returns:
            full_nacs_score: Integer score in [0, 100] under Configuration D (smoothed NACS).
        """
        s_size = compute_raw_score(client_size, all_client_sizes, loss_before, loss_after, config="size_only")
        s_sq = compute_raw_score(client_size, all_client_sizes, loss_before, loss_after, config="size_quality")
        s_sqn = compute_raw_score(client_size, all_client_sizes, loss_before, loss_after, config="size_quality_noise", noise_multiplier=noise_multiplier)

        # Configuration D: Combined raw score fed into EMA smoothing across rounds
        s_qual = score_from_quality(loss_before, loss_after)
        raw_comb = (s_size + s_qual) / 2.0
        trust_weight = 1.0 / (1.0 + float(noise_multiplier))

        if client_id not in self.history or len(self.history[client_id]) == 0:
            nacs_score = float(raw_comb)
        else:
            previous_nacs_score = float(self.history[client_id][-1])
            nacs_score = trust_weight * raw_comb + (1.0 - trust_weight) * previous_nacs_score

        clipped_nacs_score = int(np.clip(round(nacs_score), 0, 100))

        if client_id not in self.history:
            self.history[client_id] = []
            self.config_history[client_id] = []

        self.history[client_id].append(clipped_nacs_score)
        self.config_history[client_id].append({
            "size_only": s_size,
            "size_quality": s_sq,
            "size_quality_noise": s_sqn,
            "full_nacs": clipped_nacs_score,
        })

        return clipped_nacs_score

    def get_latest_scores(self, client_id: int) -> Dict[str, int]:
        """
        Returns the latest round's 4-configuration scores for a given client.
        """
        if client_id in self.config_history and self.config_history[client_id]:
            return self.config_history[client_id][-1]
        return {"size_only": 0, "size_quality": 0, "size_quality_noise": 0, "full_nacs": 0}

    def get_score_variance(self, client_id: int, config: str = "full_nacs") -> float:
        """
        Returns the variance of a client's scores across all recorded rounds for a specified configuration.
        """
        if client_id not in self.config_history or not self.config_history[client_id]:
            return 0.0
        scores = [h[config] for h in self.config_history[client_id] if config in h]
        if not scores:
            return 0.0
        return float(np.var(scores))
