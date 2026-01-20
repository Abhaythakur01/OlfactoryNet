"""Training loop and utilities for odor prediction models."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..models.heads import OdorPredictor
from .metrics import OdorMetrics, find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100

    # Scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Loss
    use_class_weights: bool = False
    pos_weight: Optional[torch.Tensor] = None
    loss_type: str = "bce"  # "bce" or "focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None

    # Misc
    num_workers: int = 0
    pin_memory: bool = True
    gradient_clip_val: Optional[float] = 1.0

    # Logging
    use_wandb: bool = False
    wandb_project: str = "olfactorynet"
    wandb_run_name: Optional[str] = None


@dataclass
class TrainingState:
    """State of the training process."""

    epoch: int = 0
    best_val_auroc: float = 0.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_aurocs: list[float] = field(default_factory=list)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced multi-label classification.

    Focuses training on hard examples by down-weighting easy ones.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """Initialize Focal Loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Balancing factor for positive class
            pos_weight: Per-class positive weights
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits (N, C)
            targets: Binary targets (N, C)
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_weight * loss

        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)
            weight = torch.where(targets == 1, pos_weight, torch.ones_like(pos_weight))
            loss = weight * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WandbLogger:
    """Weights & Biases logger wrapper."""

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Initialize wandb logging.

        Args:
            project: W&B project name
            run_name: Optional run name
            config: Configuration dict to log
        """
        self.enabled = False
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                reinit=True,
            )
            self.enabled = True
            logger.info(f"Wandb initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics."""
        if self.enabled:
            for key, value in metrics.items():
                self.wandb.run.summary[key] = value

    def finish(self) -> None:
        """Finish wandb run."""
        if self.enabled:
            self.wandb.finish()


class Trainer:
    """Trainer for odor prediction models."""

    def __init__(
        self,
        model: OdorPredictor,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        wandb_config: Optional[dict] = None,
    ):
        """Initialize trainer.

        Args:
            model: OdorPredictor model
            config: Training configuration
            device: Device to train on
            wandb_config: Full config dict for wandb logging
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self.model.to(self.device)

        # Loss function
        if config.pos_weight is not None:
            pos_weight = config.pos_weight.to(self.device)
        else:
            pos_weight = None

        if config.loss_type == "focal":
            self.criterion = FocalLoss(
                gamma=config.focal_gamma,
                alpha=config.focal_alpha,
                pos_weight=pos_weight,
            )
            logger.info(f"Using Focal Loss (gamma={config.focal_gamma})")
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info("Using BCE Loss")

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Maximize AUROC
            patience=config.lr_patience,
            factor=config.lr_factor,
            min_lr=config.min_lr,
        )

        # Metrics
        self.metrics = OdorMetrics(
            num_labels=model.num_labels, device=self.device
        )

        # State
        self.state = TrainingState()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb logging
        self.wandb_logger: Optional[WandbLogger] = None
        if config.use_wandb:
            self.wandb_logger = WandbLogger(
                project=config.wandb_project,
                run_name=config.wandb_run_name,
                config=wandb_config,
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Run full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader for final evaluation

        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.max_epochs):
            self.state.epoch = epoch

            # Train epoch
            train_loss = self._train_epoch(train_loader)
            self.state.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            self.state.val_losses.append(val_loss)

            val_auroc = val_metrics["auroc_macro"]
            self.state.val_aurocs.append(val_auroc)

            # Update scheduler
            self.scheduler.step(val_auroc)

            # Log progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val AUROC: {val_auroc:.4f}, "
                f"LR: {current_lr:.2e}"
            )

            # Log to wandb
            if self.wandb_logger:
                log_metrics = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/auroc_macro": val_auroc,
                    "val/auroc_micro": val_metrics["auroc_micro"],
                    "val/f1_macro": val_metrics["f1_macro"],
                    "val/f1_micro": val_metrics["f1_micro"],
                    "val/precision_macro": val_metrics["precision_macro"],
                    "val/recall_macro": val_metrics["recall_macro"],
                    "learning_rate": current_lr,
                }
                self.wandb_logger.log(log_metrics, step=epoch)

            # Check for improvement
            if val_auroc > self.state.best_val_auroc + self.config.early_stopping_min_delta:
                self.state.best_val_auroc = val_auroc
                self.state.best_epoch = epoch
                self.state.epochs_without_improvement = 0

                # Save best model
                self._save_checkpoint("best_model.pt")
                logger.info(f"  New best model! AUROC: {val_auroc:.4f}")
            else:
                self.state.epochs_without_improvement += 1

            # Early stopping
            if self.state.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(best epoch: {self.state.best_epoch+1})"
                )
                break

        # Load best model for final evaluation
        self._load_checkpoint("best_model.pt")

        # Final validation metrics
        _, final_val_metrics = self._validate(val_loader)

        results = {
            "best_epoch": self.state.best_epoch + 1,
            "best_val_auroc": self.state.best_val_auroc,
            "final_val_metrics": final_val_metrics,
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
            "val_aurocs": self.state.val_aurocs,
        }

        # Test evaluation if provided
        if test_loader is not None:
            test_loss, test_metrics = self._validate(test_loader)
            results["test_loss"] = test_loss
            results["test_metrics"] = test_metrics
            logger.info(f"Test AUROC: {test_metrics['auroc_macro']:.4f}")

            # Log test metrics to wandb
            if self.wandb_logger:
                self.wandb_logger.log_summary({
                    "best_epoch": results["best_epoch"],
                    "best_val_auroc": results["best_val_auroc"],
                    "test/auroc_macro": test_metrics["auroc_macro"],
                    "test/f1_macro": test_metrics["f1_macro"],
                })

        # Finish wandb run
        if self.wandb_logger:
            self.wandb_logger.finish()

        return results

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(loader, desc=f"Epoch {self.state.epoch+1}", leave=False)

        for batch in progress:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model.forward_batch(batch)
            loss = self.criterion(logits, batch.y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress.set_postfix({"loss": loss.item()})

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, dict]:
        """Validate model.

        Returns:
            Tuple of (average loss, metrics dict)
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)

            logits = self.model.forward_batch(batch)
            loss = self.criterion(logits, batch.y)

            # Convert logits to probabilities for metrics
            probs = torch.sigmoid(logits)
            self.metrics.update(probs, batch.y)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.state.epoch,
            "best_val_auroc": self.state.best_val_auroc,
        }, path)

    def _load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")

    def overfit_single_batch(
        self,
        loader: DataLoader,
        num_steps: int = 100,
    ) -> list[float]:
        """Sanity check: overfit on a single batch.

        Args:
            loader: Data loader
            num_steps: Number of optimization steps

        Returns:
            List of losses
        """
        self.model.train()

        # Get single batch
        batch = next(iter(loader)).to(self.device)

        losses = []
        for step in range(num_steps):
            self.optimizer.zero_grad()
            logits = self.model.forward_batch(batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if step % 10 == 0:
                logger.info(f"Step {step}: Loss = {loss.item():.4f}")

        logger.info(f"Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
        return losses
