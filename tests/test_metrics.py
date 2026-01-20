"""Tests for metrics module."""

import pytest
import torch

from olfactorynet.training.metrics import (
    OdorMetrics,
    find_optimal_threshold,
    compute_label_frequencies,
)


class TestOdorMetrics:
    """Tests for OdorMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create metrics for testing."""
        return OdorMetrics(num_labels=5, threshold=0.5)

    def test_init(self, metrics):
        """Test metrics initialization."""
        assert metrics.num_labels == 5
        assert metrics.threshold == 0.5

    def test_update_and_compute(self, metrics):
        """Test updating and computing metrics."""
        # Create sample predictions and targets
        preds = torch.tensor([
            [0.9, 0.1, 0.8, 0.2, 0.5],
            [0.2, 0.9, 0.3, 0.8, 0.1],
        ])
        targets = torch.tensor([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
        ])

        metrics.update(preds, targets)
        results = metrics.compute()

        # Check that all expected metrics are present
        assert "auroc_macro" in results
        assert "auroc_micro" in results
        assert "f1_macro" in results
        assert "f1_micro" in results
        assert "precision_macro" in results
        assert "recall_macro" in results
        assert "hamming" in results

        # All metrics should be floats
        for value in results.values():
            assert isinstance(value, float)

    def test_perfect_predictions(self, metrics):
        """Test with perfect predictions."""
        # Need more samples to compute AUROC reliably
        preds = torch.tensor([
            [0.9, 0.1, 0.9, 0.1, 0.9],
            [0.1, 0.9, 0.1, 0.9, 0.1],
            [0.9, 0.1, 0.9, 0.1, 0.9],
            [0.1, 0.9, 0.1, 0.9, 0.1],
        ])
        targets = torch.tensor([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ])

        metrics.update(preds, targets)
        results = metrics.compute()

        # With perfect predictions, F1 should be high
        # AUROC may not be computable for all labels with small sample sizes
        assert results["f1_macro"] > 0.9

    def test_reset(self, metrics):
        """Test metrics reset."""
        preds = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.5]])
        targets = torch.tensor([[1, 0, 1, 0, 0]])

        metrics.update(preds, targets)
        metrics.reset()

        # After reset, need new data
        preds2 = torch.tensor([[0.1, 0.9, 0.1, 0.9, 0.9]])
        targets2 = torch.tensor([[0, 1, 0, 1, 1]])

        metrics.update(preds2, targets2)
        results = metrics.compute()

        # Should compute on new data only
        assert "auroc_macro" in results

    def test_compute_per_label_auroc(self, metrics):
        """Test per-label AUROC computation."""
        preds = torch.tensor([
            [0.9, 0.1, 0.8, 0.2, 0.5],
            [0.2, 0.9, 0.3, 0.8, 0.1],
            [0.8, 0.2, 0.9, 0.1, 0.6],
        ])
        targets = torch.tensor([
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
        ])

        metrics.update(preds, targets)
        per_label = metrics.compute_per_label_auroc()

        assert per_label.shape == (5,)

    def test_to_device(self, metrics):
        """Test moving metrics to device."""
        metrics_moved = metrics.to(torch.device("cpu"))
        assert metrics_moved.device == torch.device("cpu")


class TestFindOptimalThreshold:
    """Tests for optimal threshold finding."""

    def test_find_optimal_threshold(self):
        """Test finding optimal threshold."""
        preds = torch.tensor([
            [0.9, 0.1, 0.8],
            [0.2, 0.9, 0.3],
            [0.7, 0.3, 0.6],
        ])
        targets = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ])

        threshold, f1 = find_optimal_threshold(preds, targets)

        assert 0.1 <= threshold <= 0.9
        assert 0.0 <= f1 <= 1.0

    def test_threshold_range(self):
        """Test that threshold is within expected range."""
        # Create data where optimal threshold should be around 0.5
        preds = torch.rand(100, 10)
        targets = (preds > 0.5).float()

        threshold, _ = find_optimal_threshold(preds, targets)

        # Threshold should be reasonable
        assert 0.3 <= threshold <= 0.7


class TestComputeLabelFrequencies:
    """Tests for label frequency computation."""

    def test_compute_frequencies(self):
        """Test computing label frequencies."""
        targets = torch.tensor([
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=torch.float)

        freqs = compute_label_frequencies(targets)

        assert freqs.shape == (5,)
        assert freqs[0] == 0.75  # Label 0 appears in 3/4 samples
        assert freqs[4] == 0.25  # Label 4 appears in 1/4 samples

    def test_all_ones(self):
        """Test with all positive labels."""
        targets = torch.ones(10, 5)
        freqs = compute_label_frequencies(targets)

        assert torch.allclose(freqs, torch.ones(5))

    def test_all_zeros(self):
        """Test with all negative labels."""
        targets = torch.zeros(10, 5)
        freqs = compute_label_frequencies(targets)

        assert torch.allclose(freqs, torch.zeros(5))
