#!/usr/bin/env python3
"""
model_training_module.py

Refactor of experiment7 model_training.py:
- moves main() logic into run_training(...)
- keeps helper save_training_results(...)
- no CLI parsing here (that goes in run_training.py)
"""

from __future__ import annotations
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

print("importing libraries")

import project1_a.scripts.models.autoencoder_fitting as autoencoder_fitting
from project1_a.scripts.models.models import Conv1DAutoencoder
from project1_a.scripts.models.early_stopping import EarlyStopping
import project1_a.scripts.utils as utils
from project1_a.scripts.utils import save_training_results


def run_training(
    *,
    config: str | None = None,
    timestamp: str | None = None,
    run_index: int = 0,
    model_selection_timestamp: str | None = None,
    exp_dir: Path,
):
    """
    Train AE models using experiment7 config layout.
    """
    print("Starting run_training()")

    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = "model_training"

    AE_config_path = "general/general_AE_config.yaml"
    AE_config = utils.load_config(config_name=AE_config_path)

    if config is None:
        cfg = AE_config
    else:
        model_training_config = utils.load_config(config_name=config)
        cfg = utils.merge_configs(AE_config, model_training_config)

    data_config = utils.load_config(config_name= "filenames.yaml")

    validation_percentage = cfg["training"]["validation_data_percentage"]

    batch_cfg = cfg.get("batch_training", {})
    if batch_cfg.get("enabled", False):
        train_batch_size = batch_cfg.get("train_batch_size")
        test_batch_size = batch_cfg.get("test_batch_size")
        train_shuffle = batch_cfg.get("train_shuffle")
        test_shuffle = batch_cfg.get("test_shuffle")
        batch_training = True
    else:
        batch_training = False
        train_shuffle = False
        test_shuffle = False
        train_batch_size = None
        test_batch_size = None

    # Seed
    if cfg.get("random_seed") and cfg["random_seed"].get("base_seed") is not None:
        seed = cfg["random_seed"]["base_seed"]
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
    else:
        print("No random seed set (non-deterministic run).")
        seed = None

    running_multi_iterations = run_index > 0

    results_dir, logs_dir = utils.setup_output_directories(
        exp_dir, timestamp, script=script_name, running_multi_iterations=False
    )
    log_file_path = utils.setup_logging(logs_dir, script=script_name, timestamp=timestamp)

    params_file = utils.write_experiment_parameters(
        cfg,
        results_dir,
        timestamp,
        args=None,
        experiment_name="Model Training",
        description="Train autoencoder models on training data using parameters from config file",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    training_data_filename = data_config["model_training"]["training_filename"]
    best_segments_train = utils.load_training_data(exp_dir, training_data_filename)

    standardisation_method = cfg["training"]["standardisation_method"]
    training_data, validation_data = autoencoder_fitting.prepare_training_val_tensors_and_standardise(
        best_segments_train,
        standardisation_method,
        validation_percentage,
    )

    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")

    training_data = training_data.to(device)
    validation_data = validation_data.to(device)

    training_meta_data = {}

    for model_name in cfg["models"].keys():
        print(f"\n\n******************* Training {model_name} model *******************")

        if model_selection_timestamp is not None:
            learning_rate, num_layers, encoding_dim = autoencoder_fitting.load_best_model_params(
                model_selection_timestamp,
                model_name,
                model_selection_dir=exp_dir,
                is_age_model=False,
            )
        else:
            model_config = cfg["models"][model_name]
            learning_rate = model_config["learning_rate"]
            num_layers = model_config["num_layers"]
            encoding_dim = model_config["encoding_dim"]

        print(
            f"Using parameters for {model_name} - "
            f"Learning Rate: {learning_rate}, Num Layers: {num_layers}, Encoding Dim: {encoding_dim}"
        )

        # Update params file
        with open(params_file, "r") as f:
            experiment_params = yaml.safe_load(f)

        experiment_params["model_parameters"] = {
            model_name: {
                "model_selection_filename": model_selection_timestamp if model_selection_timestamp else None,
                "learning_rate": learning_rate,
                "num_layers": num_layers,
                "encoding_dim": encoding_dim,
            }
        }

        with open(params_file, "w") as f:
            yaml.dump(experiment_params, f, default_flow_style=False, indent=2)

        # Create model
        if model_name == "Conv1D":
            model = Conv1DAutoencoder(input_channels=2, encoding_dim=encoding_dim, num_layers=num_layers).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=5, delta=0.01)

        prep_training_data = autoencoder_fitting.prepare_data_for_model(model_name, training_data, device=device)
        prep_validation_data = autoencoder_fitting.prepare_data_for_model(model_name, validation_data, device=device)
        print(f"Prepared data shape for {model_name}: {prep_training_data.shape}")

        if batch_training:
            print(f"Using batch training with batch size: {train_batch_size}, shuffle: {train_shuffle}")
        else:
            print("Using full-batch training")

        train_dataset = TensorDataset(prep_training_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size if batch_training else len(train_dataset),
            shuffle=train_shuffle,
        )

        validation_dataset = TensorDataset(prep_validation_data)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=test_batch_size if batch_training else len(validation_dataset),
            shuffle=test_shuffle,
        )

        training_losses, stopped_epoch = autoencoder_fitting.train_AE_batch_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            epochs=cfg["training"]["epochs"],
            val_loader=validation_loader,
            early_stopping=early_stopping,
        )

        # Final metrics + latents
        model.eval()
        with torch.no_grad():
            final_reconstructed_train, latent_variable_train, _ = model(prep_training_data)
            final_loss_train = criterion(final_reconstructed_train, prep_training_data)
            final_mse_train = F.mse_loss(final_reconstructed_train, prep_training_data, reduction="none").mean(dim=-1)
            final_mae_train = F.l1_loss(final_reconstructed_train, prep_training_data, reduction="none").mean(dim=-1)

            final_reconstructed_val, latent_variables_validation, _ = model(prep_validation_data)
            final_loss_val = criterion(final_reconstructed_val, prep_validation_data)
            final_mse_val = F.mse_loss(final_reconstructed_val, prep_validation_data, reduction="none").mean(dim=-1)
            final_mae_val = F.l1_loss(final_reconstructed_val, prep_validation_data, reduction="none").mean(dim=-1)

        training_meta_data[model_name] = {
            "training_losses": training_losses["train"],
            "validation_losses": training_losses["val"],
            "training_metrics": {
                "final_loss": final_loss_train.item(),
                "final_mse": final_mse_train,
                "final_mae": final_mae_train,
            },
            "validation_metrics": {
                "final_loss": final_loss_val.item(),
                "final_mse": final_mse_val,
                "final_mae": final_mae_val,
            },
            "epochs": cfg["training"]["epochs"],
            "stopped_epoch": stopped_epoch,
            "learning_rate": learning_rate,
            "architecture": {"num_layers": num_layers, "encoding_dim": encoding_dim},
            "latent_variables": {"train": latent_variable_train, "validation": latent_variables_validation},
        }

        print(f"{model_name} training completed - Final Loss: {final_loss_train.item():.6f}")

        model_save_path = results_dir / f"{model_name}_trained_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "encoding_dim": encoding_dim,
                "num_layers": num_layers,
                "optimizer_state": optimizer.state_dict(),
                "learning_rate": learning_rate,
            },
            model_save_path,
        )
        print(f"{model_name} model saved to: {model_save_path}")

    print("\n\nSaving training results...")
    meta_data_path = save_training_results(training_meta_data, results_dir, running_multi_iterations, run_index, filename_prefix="experiment7_02_model_training_meta_data")

    print("\nModel training completed successfully!")
    print(f"Results saved in: {meta_data_path}")
    print(f"Models saved in: {results_dir}")
    print(f"Log file: {log_file_path}")

    return results_dir
