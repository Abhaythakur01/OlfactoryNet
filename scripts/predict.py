#!/usr/bin/env python
"""Inference script for predicting odor descriptors from SMILES."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Batch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from olfactorynet.data.featurizer import MolecularFeaturizer
from olfactorynet.models import GCNModel, GATModel, GINModel, MPNNModel, OdorPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    metadata_path: str,
    config_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[OdorPredictor, list[str], MolecularFeaturizer]:
    """Load trained model and metadata.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        metadata_path: Path to dataset metadata (.pkl file)
        config_path: Optional path to config JSON for model architecture
        device: Device to load model on

    Returns:
        Tuple of (model, label_names, featurizer)
    """
    # Load metadata
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    label_names = metadata["label_names"]
    atom_dim = metadata["atom_dim"]
    bond_dim = metadata["bond_dim"]
    num_labels = len(label_names)

    # Load config if provided
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
        model_config = config.get("config", {}).get("model", {})
    else:
        # Default to GCN with standard settings
        model_config = {
            "type": "gcn",
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.2,
            "pooling": "mean",
        }

    # Create model based on config
    model_type = model_config.get("type", "gcn")

    if model_type == "gcn":
        backbone = GCNModel(
            atom_dim=atom_dim,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.2),
            pooling=model_config.get("pooling", "mean"),
        )
    elif model_type == "gat":
        edge_dim = bond_dim if model_config.get("use_edge_features", True) else None
        backbone = GATModel(
            atom_dim=atom_dim,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.2),
            pooling=model_config.get("pooling", "mean"),
            num_heads=model_config.get("num_heads", 4),
            edge_dim=edge_dim,
        )
    elif model_type == "gin":
        edge_dim = bond_dim if model_config.get("use_edge_features", True) else None
        backbone = GINModel(
            atom_dim=atom_dim,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.2),
            pooling=model_config.get("pooling", "mean"),
            edge_dim=edge_dim,
        )
    elif model_type == "mpnn":
        backbone = MPNNModel(
            atom_dim=atom_dim,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.2),
            pooling=model_config.get("pooling", "set2set"),
            edge_dim=bond_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    head_config = model_config.get("head", {})
    model = OdorPredictor(
        backbone=backbone,
        num_labels=num_labels,
        head_hidden_dim=head_config.get("hidden_dim", 256),
        head_dropout=head_config.get("dropout", 0.2),
        head_num_layers=head_config.get("num_layers", 1),
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create featurizer
    featurizer = MolecularFeaturizer()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model type: {model_type}, Labels: {num_labels}")

    return model, label_names, featurizer


def predict_single(
    smiles: str,
    model: OdorPredictor,
    label_names: list[str],
    featurizer: MolecularFeaturizer,
    device: torch.device,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> dict:
    """Predict odor descriptors for a single molecule.

    Args:
        smiles: SMILES string
        model: Trained model
        label_names: List of odor descriptor names
        featurizer: Molecular featurizer
        device: Device
        threshold: Probability threshold for positive prediction
        top_k: Return top K predictions (None for all above threshold)

    Returns:
        Dictionary with predictions
    """
    # Featurize
    data = featurizer.featurize(smiles)
    if data is None:
        return {"error": f"Failed to parse SMILES: {smiles}", "smiles": smiles}

    # Create batch
    batch = Batch.from_data_list([data]).to(device)

    # Predict
    with torch.no_grad():
        probs = model.predict_proba(batch)[0].cpu().numpy()

    # Get predictions
    predictions = []
    for idx, (name, prob) in enumerate(zip(label_names, probs)):
        predictions.append({
            "label": name,
            "probability": float(prob),
            "predicted": bool(prob >= threshold),
        })

    # Sort by probability
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Filter
    if top_k is not None:
        predictions = predictions[:top_k]
    else:
        predictions = [p for p in predictions if p["predicted"]]

    return {
        "smiles": smiles,
        "predictions": predictions,
        "num_predicted": sum(1 for p in predictions if p["predicted"]),
    }


def predict_batch(
    smiles_list: list[str],
    model: OdorPredictor,
    label_names: list[str],
    featurizer: MolecularFeaturizer,
    device: torch.device,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Predict odor descriptors for multiple molecules.

    Args:
        smiles_list: List of SMILES strings
        model: Trained model
        label_names: List of odor descriptor names
        featurizer: Molecular featurizer
        device: Device
        threshold: Probability threshold
        top_k: Return top K predictions per molecule

    Returns:
        List of prediction dictionaries
    """
    results = []
    valid_data = []
    valid_indices = []

    # Featurize all molecules
    for idx, smiles in enumerate(smiles_list):
        data = featurizer.featurize(smiles)
        if data is None:
            results.append({
                "error": f"Failed to parse SMILES: {smiles}",
                "smiles": smiles,
            })
        else:
            valid_data.append(data)
            valid_indices.append(idx)
            results.append(None)  # Placeholder

    if not valid_data:
        return results

    # Batch prediction
    batch = Batch.from_data_list(valid_data).to(device)

    with torch.no_grad():
        all_probs = model.predict_proba(batch).cpu().numpy()

    # Process predictions
    for i, (idx, probs) in enumerate(zip(valid_indices, all_probs)):
        smiles = smiles_list[idx]

        predictions = []
        for name, prob in zip(label_names, probs):
            predictions.append({
                "label": name,
                "probability": float(prob),
                "predicted": bool(prob >= threshold),
            })

        predictions.sort(key=lambda x: x["probability"], reverse=True)

        if top_k is not None:
            predictions = predictions[:top_k]
        else:
            predictions = [p for p in predictions if p["predicted"]]

        results[idx] = {
            "smiles": smiles,
            "predictions": predictions,
            "num_predicted": sum(1 for p in predictions if p["predicted"]),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict odor descriptors for molecules"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/metadata.pkl",
        help="Path to dataset metadata",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training results JSON (for model config)",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        nargs="+",
        help="SMILES string(s) to predict",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File with SMILES (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive prediction",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Return top K predictions per molecule",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cpu, cuda, or auto)",
    )

    args = parser.parse_args()

    # Get device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model, label_names, featurizer = load_model(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        config_path=args.config,
        device=device,
    )

    # Get SMILES to predict
    smiles_list = []
    if args.smiles:
        smiles_list.extend(args.smiles)
    if args.input_file:
        with open(args.input_file, "r") as f:
            smiles_list.extend(line.strip() for line in f if line.strip())

    if not smiles_list:
        parser.error("No SMILES provided. Use --smiles or --input-file")

    # Predict
    logger.info(f"Predicting for {len(smiles_list)} molecule(s)...")

    if len(smiles_list) == 1:
        results = predict_single(
            smiles_list[0],
            model,
            label_names,
            featurizer,
            device,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    else:
        results = predict_batch(
            smiles_list,
            model,
            label_names,
            featurizer,
            device,
            threshold=args.threshold,
            top_k=args.top_k,
        )

    # Output
    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        logger.info(f"Results saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
