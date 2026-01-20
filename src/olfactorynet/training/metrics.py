"""Multi-label classification metrics for odor prediction."""

from __future__ import annotations

from typing import Optional

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelHammingDistance,
)


class OdorMetrics:
    """Collection of metrics for multi-label odor classification.

    Uses torchmetrics for efficient, batched metric computation.
    """

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """Initialize metrics.

        Args:
            num_labels: Number of odor labels
            threshold: Probability threshold for binary predictions
            device: Device to place metrics on
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.device = device or torch.device("cpu")

        # Create metric collection
        self.metrics = MetricCollection({
            # AUROC - primary metric, threshold-independent
            "auroc_macro": MultilabelAUROC(
                num_labels=num_labels, average="macro", thresholds=None
            ),
            "auroc_micro": MultilabelAUROC(
                num_labels=num_labels, average="micro", thresholds=None
            ),
            # F1 Score
            "f1_macro": MultilabelF1Score(
                num_labels=num_labels, average="macro", threshold=threshold
            ),
            "f1_micro": MultilabelF1Score(
                num_labels=num_labels, average="micro", threshold=threshold
            ),
            # Precision and Recall
            "precision_macro": MultilabelPrecision(
                num_labels=num_labels, average="macro", threshold=threshold
            ),
            "recall_macro": MultilabelRecall(
                num_labels=num_labels, average="macro", threshold=threshold
            ),
            # Hamming distance (lower is better)
            "hamming": MultilabelHammingDistance(
                num_labels=num_labels, threshold=threshold
            ),
        }).to(self.device)

        # Per-label AUROC for analysis
        self.per_label_auroc = MultilabelAUROC(
            num_labels=num_labels, average=None, thresholds=None
        ).to(self.device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with a batch of predictions.

        Args:
            preds: Predicted probabilities (batch_size, num_labels)
            targets: Ground truth labels (batch_size, num_labels)
        """
        # Ensure tensors are on correct device
        preds = preds.to(self.device)
        targets = targets.to(self.device).long()

        # Update all metrics
        self.metrics.update(preds, targets)
        self.per_label_auroc.update(preds, targets)

    def compute(self) -> dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        results = self.metrics.compute()

        # Convert to Python floats
        return {name: value.item() for name, value in results.items()}

    def compute_per_label_auroc(self) -> torch.Tensor:
        """Compute per-label AUROC scores.

        Returns:
            Tensor of shape (num_labels,) with AUROC per label
        """
        return self.per_label_auroc.compute()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()
        self.per_label_auroc.reset()

    def to(self, device: torch.device) -> "OdorMetrics":
        """Move metrics to device."""
        self.device = device
        self.metrics = self.metrics.to(device)
        self.per_label_auroc = self.per_label_auroc.to(device)
        return self


def find_optimal_threshold(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 50,
) -> tuple[float, float]:
    """Find optimal classification threshold based on F1 score.

    Args:
        preds: Predicted probabilities (N, num_labels)
        targets: Ground truth labels (N, num_labels)
        num_thresholds: Number of thresholds to try

    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = torch.linspace(0.1, 0.9, num_thresholds)

    for threshold in thresholds:
        # Binary predictions
        binary_preds = (preds >= threshold).float()

        # Calculate F1
        tp = (binary_preds * targets).sum()
        fp = (binary_preds * (1 - targets)).sum()
        fn = ((1 - binary_preds) * targets).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1.item()
            best_threshold = threshold.item()

    return best_threshold, best_f1


def compute_label_frequencies(targets: torch.Tensor) -> torch.Tensor:
    """Compute frequency of each label in the dataset.

    Args:
        targets: All target labels (N, num_labels)

    Returns:
        Tensor of shape (num_labels,) with frequencies
    """
    return targets.sum(dim=0) / targets.shape[0]
