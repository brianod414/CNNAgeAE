#!/usr/bin/env python3
"""
fine_tuning_module.py

Refactor of fine_tuning.py:
- main() logic moved into run_fine_tuning(...)
- no argparse here (CLI wrapper is separate)
"""

from __future__ import annotations
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from project1_a.scripts.models.early_stopping import EarlyStopping
from project1_a.scripts.models.models import Conv1DAutoencoder
import project1_a.scripts.models.autoencoder_fitting as autoencoder_fitting
import project1_a.scripts.utils as utils
from project1_a.scripts.utils import save_training_results


def run_fine_tuning(
    *,
    config: str = "stages/fine_tuning_config.yaml",
    run_index: int = 0,
    timestamp: str | None = None,
    models_timestamp: str,
    model_selection_timestamp: str | None = None,
    use_config_lambda: bool = False,
    exp_dir: Path
):
    """
    Fine-tune pretrained AEs with age regression head.

    Minimal-change version: still uses experiment7 folder conventions.
    """
    print("Starting run_fine_tuning()")

    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = "fine_tuning"

    print(f"Starting {script_name} with config: {config} and timestamp: {timestamp}")

    results_dir, logs_dir = utils.setup_output_directories(
        exp_dir, timestamp, script=script_name, running_multi_iterations=False
    )
    utils.setup_logging(logs_dir, script_name, timestamp)

    # Load configuration
    stage_cfg = utils.load_config(config_name=config)
    AE_config = utils.load_config(config_name="general/general_AE_config.yaml")
    cfg = utils.merge_configs(AE_config, stage_cfg)

    data_config = utils.load_config(config_name="filenames.yaml")

    # record keeping
    cfg["pre_trained_model_timestamp"] = models_timestamp
    cfg["model_selection_timestamp"] = model_selection_timestamp
    cfg["use_config_lambda"] = bool(use_config_lambda)

    # Batch config
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

    # parameters YAML
    params_file = utils.write_experiment_parameters(
        cfg,
        results_dir,
        timestamp,
        args=None,
        experiment_name="Fine tuning",
        description="Fine tune autoencoder models with age regression head",
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    training_data_filename = data_config["fine_tuning"]["training_filename"]
    best_segments_train = utils.load_training_data(exp_dir, training_data_filename)

    # tensors + standardisation
    standardisation_method = cfg["training"]["standardisation_method"]
    validation_percentage = cfg["training"]["validation_data_percentage"]

    training_data, validation_data, training_ids, validation_ids = (
        autoencoder_fitting.prepare_training_val_tensors_and_standardise(
            best_segments_train,
            standardisation_method,
            validation_percentage,
            return_ids=True,
        )
    )
    print(f"Combined training data shape: {training_data.shape}")

    training_data = training_data.to(device)
    validation_data = validation_data.to(device)

    training_meta_data = {}

    for model_name in cfg["models"].keys():
        print(f"\n\n******************* Training {model_name} model *******************")

        pre_trained_model_path = (
            exp_dir
            / "results"
            / f"run_{cfg['pre_trained_model_timestamp']}"
            / "model_training"
            / f"{model_name}_trained_model.pth"
        )
        checkpoint = torch.load(pre_trained_model_path, map_location=device)
        encoding_dim = checkpoint["encoding_dim"]
        num_layers = checkpoint["num_layers"]
        ae_learning_rate = checkpoint.get("learning_rate", checkpoint.get("ae_learning_rate"))

        print(
            f"Loaded pre-trained {model_name} model from {pre_trained_model_path} "
            f"with encoding_dim={encoding_dim}, num_layers={num_layers}, lr={ae_learning_rate}"
        )

        if cfg["models"][model_name]["scale_ae_learning_rate"]:
            ae_learning_rate /= 10

        if cfg["models"][model_name]["scale_age_learning_rate"]:
            age_learning_rate = ae_learning_rate * 10
        else:
            age_learning_rate = cfg["models"][model_name]["age_learning_rate"]

        weight_decay = cfg["models"][model_name]["weight_decay"]

        # Choose lambda_age
        if (model_selection_timestamp is not None) and (not use_config_lambda):
            age_lambda = autoencoder_fitting.load_best_model_params(
                model_selection_timestamp,
                model_name,
                model_selection_dir=exp_dir,
                is_age_model=True,
            )
        elif use_config_lambda:
            age_lambda = cfg["loss_function"]["age_lambda"]
        else:
            raise ValueError("Either model_selection_timestamp must be provided or use_config_lambda must be True")

        # Build model w/ age regression head
        if model_name == "Conv1D":
            model = Conv1DAutoencoder(
                input_channels=2,
                encoding_dim=encoding_dim,
                num_layers=num_layers,
                use_age_regression=True,
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            # fallback (older format)
            model.load_state_dict(checkpoint, strict=False)

        # Split params into AE vs age head
        ae_params = [p for name, p in model.named_parameters() if "age_predictor" not in name]
        age_params = [p for name, p in model.named_parameters() if "age_predictor" in name]

        ae_optimizer = torch.optim.Adam(ae_params, lr=ae_learning_rate)
        age_optimizer = torch.optim.Adam(age_params, lr=age_learning_rate, weight_decay=weight_decay)

        ae_criterion = nn.MSELoss()
        age_criterion = nn.SmoothL1Loss()

        prep_training_data = autoencoder_fitting.prepare_data_for_model(model_name, training_data, device=device)
        prep_validation_data = autoencoder_fitting.prepare_data_for_model(model_name, validation_data, device=device)
        print(f"Prepared data shape for {model_name}: {prep_training_data.shape}")

        # Ages
        training_ages = utils.get_normalised_patient_ages(training_ids)
        validation_ages = utils.get_normalised_patient_ages(validation_ids)

        training_ages_tensor = torch.tensor(list(training_ages), dtype=torch.float32, device=device)
        validation_ages_tensor = torch.tensor(list(validation_ages), dtype=torch.float32, device=device)

        early_stopping = EarlyStopping(patience=5, delta=0.01)

        if batch_training:
            print(f"Using batch training with batch size: {train_batch_size}, shuffle: {train_shuffle}")
        else:
            print("Using full-batch training")

        train_dataset = autoencoder_fitting.SignalAgeDataset(prep_training_data, training_ages_tensor)
        validation_dataset = autoencoder_fitting.SignalAgeDataset(prep_validation_data, validation_ages_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size if batch_training else len(train_dataset),
            shuffle=train_shuffle,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=test_batch_size if batch_training else len(validation_dataset),
            shuffle=test_shuffle,
        )

        print("Training with separate optimizers (batch mode)")
        results_dict, stopped_epoch = autoencoder_fitting.train_AE_age_batch_model(
            model=model,
            ae_optimizer=ae_optimizer,
            age_optimizer=age_optimizer,
            ae_criterion=ae_criterion,
            age_criterion=age_criterion,
            train_loader=train_loader,
            val_loader=validation_loader,
            early_stopping=early_stopping,
            lambda_age=age_lambda,
            lambda_separation=cfg["loss_function"]["separation_lambda"],
            pseudo_age=cfg["loss_function"]["pseudo_age"],
            epochs=cfg["training"]["epochs"],
            device=device,
        )

        # Final loss/metrics
        model.eval()
        with torch.no_grad():
            final_reconstructed, latent_variables, age_predictions = model(prep_training_data)
            total_loss, reconstruction_loss, age_loss, age_separation_loss, (supervised_age_loss, pseudo_age_loss) = (
                model.compute_loss(
                    x=prep_training_data,
                    z=latent_variables,
                    recon=final_reconstructed,
                    ae_criterion=ae_criterion,
                    age_criterion=age_criterion,
                    age_pred=age_predictions,
                    age_true=training_ages_tensor,
                    lambda_age=age_lambda,
                    lambda_separation=cfg["loss_function"]["separation_lambda"],
                    pseudo_age=cfg["loss_function"]["pseudo_age"],
                )
            )

            final_mse = F.mse_loss(final_reconstructed, prep_training_data, reduction="none").mean(dim=-1)
            final_mae = F.l1_loss(final_reconstructed, prep_training_data, reduction="none").mean(dim=-1)

        training_meta_data[model_name] = {
            "losses": results_dict,
            "stopped_epoch": stopped_epoch,
            "final_loss": total_loss.item(),
            "final_reconstruction_loss": reconstruction_loss.item(),
            "final_age_loss": age_loss.item() if age_loss is not None else 0.0,
            "final_age_separation_loss": age_separation_loss.item(),
            "final_mse": final_mse.mean().item(),
            "final_mae": final_mae.mean().item(),
            "epochs": cfg["training"]["epochs"],
            "ae_learning_rate": ae_learning_rate,
            "age_learning_rate": age_learning_rate,
            "encoding_dim": encoding_dim,
            "num_layers": num_layers,
            "latent_variables": latent_variables,
            "age_lambda": float(age_lambda),
            "weight_decay": float(weight_decay),
        }

        print(
            f"{model_name} training completed - Final Loss: {total_loss.item():.6f}, "
            f"Reconstruction Loss: {reconstruction_loss.item():.6f}, MSE: {final_mse.mean().item():.6f}"
        )

        # Save fine-tuned model
        model_save_path = results_dir / f"{model_name}_trained_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "encoding_dim": encoding_dim,
                "num_layers": num_layers,
                "ae_learning_rate": ae_learning_rate,
                "age_learning_rate": age_learning_rate,
                "ae_optimizer_state": ae_optimizer.state_dict(),
                "age_optimizer_state": age_optimizer.state_dict(),
                "age_lambda": float(age_lambda),
            },
            model_save_path,
        )
        print(f"{model_name} model saved to: {model_save_path}")


    print("\n\nSaving training results...")
    meta_data_path = save_training_results(training_meta_data, results_dir, running_multi_iterations, run_index, filename_prefix="experiment7_03_model_training_meta_data")

    print("\nModel training completed successfully!")
    print(f"Results saved in: {results_dir}")
    print(f"Models saved in: {results_dir}")

    return results_dir, meta_data_path
