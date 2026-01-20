#!/usr/bin/env python
"""Gradio web interface for OlfactoryNet odor prediction."""

from __future__ import annotations

import io
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import torch
from torch_geometric.data import Batch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from olfactorynet.data.featurizer import MolecularFeaturizer
from olfactorynet.models import GCNModel, GATModel, GINModel, MPNNModel, OdorPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
MODEL = None
LABEL_NAMES = None
FEATURIZER = None
DEVICE = None


def load_model(
    checkpoint_path: str,
    metadata_path: str,
    model_type: str = "gcn",
    hidden_dim: int = 256,
    num_layers: int = 3,
) -> str:
    """Load a trained model."""
    global MODEL, LABEL_NAMES, FEATURIZER, DEVICE

    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        LABEL_NAMES = metadata["label_names"]
        atom_dim = metadata["atom_dim"]
        bond_dim = metadata["bond_dim"]
        num_labels = len(LABEL_NAMES)

        # Create model
        if model_type == "gcn":
            backbone = GCNModel(
                atom_dim=atom_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        elif model_type == "gat":
            backbone = GATModel(
                atom_dim=atom_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=bond_dim,
            )
        elif model_type == "gin":
            backbone = GINModel(
                atom_dim=atom_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=bond_dim,
            )
        elif model_type == "mpnn":
            backbone = MPNNModel(
                atom_dim=atom_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=bond_dim,
            )
        else:
            return f"Unknown model type: {model_type}"

        MODEL = OdorPredictor(
            backbone=backbone,
            num_labels=num_labels,
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()

        FEATURIZER = MolecularFeaturizer()

        return f"Model loaded successfully on {DEVICE}. {num_labels} odor labels available."

    except Exception as e:
        return f"Error loading model: {str(e)}"


def predict_odors(
    smiles: str,
    threshold: float = 0.3,
    top_k: int = 10,
) -> tuple[str, pd.DataFrame, Optional[str]]:
    """Predict odor descriptors for a molecule.

    Returns:
        Tuple of (status message, predictions dataframe, molecule image)
    """
    global MODEL, LABEL_NAMES, FEATURIZER, DEVICE

    if MODEL is None:
        return "Please load a model first!", pd.DataFrame(), None

    if not smiles or not smiles.strip():
        return "Please enter a SMILES string.", pd.DataFrame(), None

    smiles = smiles.strip()

    try:
        # Generate molecule image
        mol_image = None
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                mol_image = img
        except Exception as e:
            logger.warning(f"Could not generate molecule image: {e}")

        # Featurize
        data = FEATURIZER.featurize(smiles)
        if data is None:
            return f"Invalid SMILES: {smiles}", pd.DataFrame(), mol_image

        # Predict
        batch = Batch.from_data_list([data]).to(DEVICE)
        with torch.no_grad():
            probs = MODEL.predict_proba(batch)[0].cpu().numpy()

        # Create results dataframe
        results = []
        for name, prob in zip(LABEL_NAMES, probs):
            results.append({
                "Odor Descriptor": name,
                "Probability": prob,
                "Predicted": "Yes" if prob >= threshold else "No",
            })

        df = pd.DataFrame(results)
        df = df.sort_values("Probability", ascending=False)

        # Filter to top_k
        df_display = df.head(top_k).reset_index(drop=True)

        # Format probability as percentage
        df_display["Probability"] = df_display["Probability"].apply(lambda x: f"{x:.1%}")

        num_predicted = (df["Predicted"] == "Yes").sum()
        status = f"Found {num_predicted} odor descriptors above {threshold:.0%} threshold"

        return status, df_display, mol_image

    except Exception as e:
        logger.exception("Prediction error")
        return f"Error: {str(e)}", pd.DataFrame(), None


def predict_batch(
    smiles_text: str,
    threshold: float = 0.3,
) -> tuple[str, pd.DataFrame]:
    """Predict odors for multiple molecules."""
    global MODEL, LABEL_NAMES, FEATURIZER, DEVICE

    if MODEL is None:
        return "Please load a model first!", pd.DataFrame()

    if not smiles_text or not smiles_text.strip():
        return "Please enter SMILES strings (one per line).", pd.DataFrame()

    smiles_list = [s.strip() for s in smiles_text.strip().split("\n") if s.strip()]

    if not smiles_list:
        return "No valid SMILES found.", pd.DataFrame()

    results = []

    for smiles in smiles_list:
        try:
            data = FEATURIZER.featurize(smiles)
            if data is None:
                results.append({
                    "SMILES": smiles,
                    "Top Odors": "Invalid SMILES",
                    "Count": 0,
                })
                continue

            batch = Batch.from_data_list([data]).to(DEVICE)
            with torch.no_grad():
                probs = MODEL.predict_proba(batch)[0].cpu().numpy()

            # Get top predictions
            top_indices = probs.argsort()[::-1][:5]
            top_odors = [
                f"{LABEL_NAMES[i]} ({probs[i]:.0%})"
                for i in top_indices
                if probs[i] >= threshold
            ]

            results.append({
                "SMILES": smiles[:30] + "..." if len(smiles) > 30 else smiles,
                "Top Odors": ", ".join(top_odors) if top_odors else "None above threshold",
                "Count": len(top_odors),
            })

        except Exception as e:
            results.append({
                "SMILES": smiles[:30] + "...",
                "Top Odors": f"Error: {str(e)}",
                "Count": 0,
            })

    df = pd.DataFrame(results)
    status = f"Processed {len(smiles_list)} molecules"

    return status, df


# Example molecules
EXAMPLES = [
    ["CCO", 0.3, 10],  # Ethanol
    ["CC(=O)OCC", 0.3, 10],  # Ethyl acetate
    ["c1ccccc1", 0.3, 10],  # Benzene
    ["CC(C)CC1=CC=C(C=C1)C(C)C", 0.3, 10],  # Isobutylbenzene
    ["CCCCCCCCCC=O", 0.3, 10],  # Decanal
    ["CC(=O)C", 0.3, 10],  # Acetone
    ["O=Cc1ccc(O)c(OC)c1", 0.3, 10],  # Vanillin
    ["CC(C)=CCCC(C)=CCO", 0.3, 10],  # Geraniol
]


def create_app() -> gr.Blocks:
    """Create the Gradio app."""

    with gr.Blocks(title="OlfactoryNet - Molecular Odor Prediction") as app:
        gr.Markdown(
            """
            # OlfactoryNet - Molecular Odor Prediction

            Predict odor descriptors for molecules using Graph Neural Networks.
            Enter a SMILES string to see predicted odor characteristics.
            """
        )

        with gr.Tab("Single Molecule"):
            with gr.Row():
                with gr.Column(scale=2):
                    smiles_input = gr.Textbox(
                        label="SMILES",
                        placeholder="Enter SMILES string (e.g., CCO for ethanol)",
                        lines=1,
                    )
                    with gr.Row():
                        threshold_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Probability Threshold",
                        )
                        topk_slider = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Top K Results",
                        )
                    predict_btn = gr.Button("Predict Odors", variant="primary")

                with gr.Column(scale=1):
                    mol_image = gr.Image(label="Molecule Structure", type="pil")

            status_text = gr.Textbox(label="Status", interactive=False)
            results_table = gr.DataFrame(
                label="Predicted Odor Descriptors",
                headers=["Odor Descriptor", "Probability", "Predicted"],
            )

            gr.Examples(
                examples=EXAMPLES,
                inputs=[smiles_input, threshold_slider, topk_slider],
                outputs=[status_text, results_table, mol_image],
                fn=predict_odors,
                cache_examples=False,
            )

            predict_btn.click(
                fn=predict_odors,
                inputs=[smiles_input, threshold_slider, topk_slider],
                outputs=[status_text, results_table, mol_image],
            )

        with gr.Tab("Batch Prediction"):
            gr.Markdown("Enter multiple SMILES strings, one per line.")

            batch_input = gr.Textbox(
                label="SMILES (one per line)",
                placeholder="CCO\nCC(=O)OCC\nc1ccccc1",
                lines=10,
            )
            batch_threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Probability Threshold",
            )
            batch_btn = gr.Button("Predict All", variant="primary")

            batch_status = gr.Textbox(label="Status", interactive=False)
            batch_results = gr.DataFrame(label="Results")

            batch_btn.click(
                fn=predict_batch,
                inputs=[batch_input, batch_threshold],
                outputs=[batch_status, batch_results],
            )

        with gr.Tab("Model Settings"):
            gr.Markdown(
                """
                ### Load a Trained Model

                Specify the paths to your trained model checkpoint and dataset metadata.
                """
            )

            checkpoint_input = gr.Textbox(
                label="Checkpoint Path",
                value="checkpoints/best_model.pt",
                placeholder="Path to model checkpoint (.pt file)",
            )
            metadata_input = gr.Textbox(
                label="Metadata Path",
                value="data/processed/metadata.pkl",
                placeholder="Path to dataset metadata (.pkl file)",
            )

            with gr.Row():
                model_type_dropdown = gr.Dropdown(
                    choices=["gcn", "gat", "gin", "mpnn"],
                    value="gcn",
                    label="Model Type",
                )
                hidden_dim_input = gr.Number(
                    value=256,
                    label="Hidden Dimension",
                )
                num_layers_input = gr.Number(
                    value=3,
                    label="Number of Layers",
                )

            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(label="Status", interactive=False)

            load_btn.click(
                fn=load_model,
                inputs=[
                    checkpoint_input,
                    metadata_input,
                    model_type_dropdown,
                    hidden_dim_input,
                    num_layers_input,
                ],
                outputs=load_status,
            )

        gr.Markdown(
            """
            ---
            **Note:** Load a trained model in the "Model Settings" tab before making predictions.

            Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and
            [Gradio](https://gradio.app/).
            """
        )

    return app


def main():
    """Launch the Gradio app."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch OlfactoryNet web interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Auto-load model from checkpoint",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata file",
    )

    args = parser.parse_args()

    # Auto-load model if paths provided
    if args.checkpoint and args.metadata:
        logger.info("Auto-loading model...")
        result = load_model(args.checkpoint, args.metadata)
        logger.info(result)

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
