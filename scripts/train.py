#!/usr/bin/env python
"""Training script for OlfactoryNet models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from olfactorynet.data import OdorDataset, DataSplitter
from olfactorynet.data.splits import create_dataloaders
from olfactorynet.models import GCNModel, GATModel, GINModel, MPNNModel, OdorPredictor
from olfactorynet.training import Trainer, OdorMetrics
from olfactorynet.training.trainer import TrainingConfig

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )


def get_device(cfg: DictConfig) -> torch.device:
    """Get device from config."""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(cfg: DictConfig, atom_dim: int, bond_dim: int, num_labels: int) -> OdorPredictor:
    """Create model from config."""
    model_type = cfg.model.type

    if model_type == "gcn":
        backbone = GCNModel(
            atom_dim=atom_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            pooling=cfg.model.pooling,
            batch_norm=cfg.model.batch_norm,
            improved=cfg.model.improved,
            add_self_loops=cfg.model.add_self_loops,
            normalize=cfg.model.normalize,
        )
    elif model_type == "gat":
        edge_dim = bond_dim if cfg.model.use_edge_features else None
        backbone = GATModel(
            atom_dim=atom_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            pooling=cfg.model.pooling,
            batch_norm=cfg.model.batch_norm,
            num_heads=cfg.model.num_heads,
            edge_dim=edge_dim,
            concat_heads=cfg.model.concat_heads,
            negative_slope=cfg.model.negative_slope,
            add_self_loops=cfg.model.add_self_loops,
        )
    elif model_type == "gin":
        edge_dim = bond_dim if cfg.model.use_edge_features else None
        backbone = GINModel(
            atom_dim=atom_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            pooling=cfg.model.pooling,
            batch_norm=cfg.model.batch_norm,
            edge_dim=edge_dim,
            train_eps=cfg.model.train_eps,
            mlp_layers=cfg.model.mlp_layers,
        )
    elif model_type == "mpnn":
        backbone = MPNNModel(
            atom_dim=atom_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            pooling=cfg.model.pooling,
            batch_norm=cfg.model.batch_norm,
            edge_dim=bond_dim,
            edge_hidden_dim=cfg.model.edge_hidden_dim,
            set2set_steps=cfg.model.set2set_steps,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = OdorPredictor(
        backbone=backbone,
        num_labels=num_labels,
        head_hidden_dim=cfg.model.head.hidden_dim,
        head_dropout=cfg.model.head.dropout,
        head_num_layers=cfg.model.head.num_layers,
    )

    return model


def create_training_config(cfg: DictConfig) -> TrainingConfig:
    """Create training config from Hydra config."""
    return TrainingConfig(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        max_epochs=cfg.training.max_epochs,
        lr_patience=cfg.training.lr_patience,
        lr_factor=cfg.training.lr_factor,
        min_lr=cfg.training.min_lr,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        checkpoint_dir=cfg.training.checkpoint_dir,
        save_best_only=cfg.training.save_best_only,
        use_class_weights=cfg.training.use_class_weights,
        loss_type=cfg.training.loss_type,
        focal_gamma=cfg.training.focal_gamma,
        focal_alpha=cfg.training.focal_alpha,
        gradient_clip_val=cfg.training.gradient_clip_val,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
        wandb_run_name=cfg.training.wandb_run_name or cfg.experiment,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function.

    Returns:
        Best validation AUROC (for hyperparameter optimization)
    """
    # Setup
    setup_logging(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg)

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = OdorDataset(
        root=cfg.data.root,
        min_label_count=cfg.data.min_label_count,
        add_hydrogens=cfg.data.add_hydrogens,
    )
    logger.info(f"Dataset: {dataset}")

    # Split data
    logger.info(f"Splitting data using {cfg.data.split_strategy} strategy...")
    splitter = DataSplitter(
        strategy=cfg.data.split_strategy,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_seed=cfg.seed,
    )
    splits = splitter.split(dataset)
    logger.info(f"Splits: {splits}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        splits=splits,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(
        cfg=cfg,
        atom_dim=dataset.atom_dim,
        bond_dim=dataset.bond_dim,
        num_labels=dataset.num_labels,
    )
    logger.info(f"Model: {model.backbone}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    training_config = create_training_config(cfg)

    # Add class weights if requested
    if training_config.use_class_weights:
        training_config.pos_weight = dataset.get_label_weights()
        logger.info("Using class weights for imbalanced labels")

    # Full config for wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        wandb_config=wandb_config,
    )

    # Sanity check: overfit single batch
    logger.info("Running overfitting sanity check...")
    overfit_losses = trainer.overfit_single_batch(train_loader, num_steps=50)
    if overfit_losses[-1] > overfit_losses[0] * 0.5:
        logger.warning("Model may not be learning - loss didn't decrease significantly")

    # Reset model and optimizer for actual training
    model = create_model(
        cfg=cfg,
        atom_dim=dataset.atom_dim,
        bond_dim=dataset.bond_dim,
        num_labels=dataset.num_labels,
    )
    trainer = Trainer(
        model=model,
        config=training_config,
        device=device,
        wandb_config=wandb_config,
    )

    # Train
    logger.info("Starting training...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Log results
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best epoch: {results['best_epoch']}")
    logger.info(f"Best validation AUROC: {results['best_val_auroc']:.4f}")

    if "test_metrics" in results:
        logger.info("Test metrics:")
        for name, value in results["test_metrics"].items():
            logger.info(f"  {name}: {value:.4f}")

    # Save results
    results_path = Path(cfg.training.checkpoint_dir) / "results.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        save_results = {
            "best_epoch": results["best_epoch"],
            "best_val_auroc": results["best_val_auroc"],
            "final_val_metrics": results["final_val_metrics"],
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if "test_metrics" in results:
            save_results["test_metrics"] = results["test_metrics"]
        json.dump(save_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Return metric for hyperparameter optimization
    return results["best_val_auroc"]


if __name__ == "__main__":
    main()
