#!/usr/bin/env python3
"""
main_run_module.py

Refactored pipeline orchestrator that calls run_*() functions directly instead of subprocess.
Maintains the same pipeline as main_run.py but with better integration and error handling.
"""

import datetime
import argparse
import os
from pathlib import Path
import importlib

# Load modules from numeric-named directories
def import_modules_project1_a(module_name, sub_dir):
    path = f"project1_a.scripts.{sub_dir}.{module_name}"
    module = importlib.import_module(path)
    return module

# # Stage 1: Model Order Selection
# model_selection = import_modules_project1_a("model_selection", "models")
# run_model_selection = model_selection.run_model_selection

# Stage 2: Base Autoencoder Training
model_training = import_modules_project1_a("model_training", "models")
run_training = model_training.run_training

# feature_extraction_ae = import_modules_project1_a("feature_extraction_AE", "models")
# run_feature_extraction_AE = feature_extraction_ae.run_feature_extraction_AE

# # Stage 3: Age-Aware Autoencoder
# model_selection_age = import_modules_project1_a("model_selection_AE_age", "models")
# run_model_selection_age = model_selection_age.run_model_selection

fine_tuning = import_modules_project1_a("fine_tuning", "models")
run_fine_tuning = fine_tuning.run_fine_tuning

# feature_extraction_age = import_modules_project1_a("feature_extraction_AE_age", "models")
# run_feature_extraction_AE_age = feature_extraction_age.run_feature_extraction_AE_age

# Stage 4: Benchmarks
pca_module = import_modules_project1_a("PCA", "benchmarks")
run_pca_benchmark = pca_module.run_pca_benchmark

summary_stats = import_modules_project1_a("summary_statistics", "benchmarks")
run_summary_statistics = summary_stats.run_summary_statistics




def main():
    parser = argparse.ArgumentParser(description="Main pipeline orchestrator (modular version)")
    parser.add_argument("--experiment_name", type=str, default="experiment7", help="Name of the experiment (default: experiment7)")
    parser.add_argument("--timestamp", type=str, help="Reuse an existing timestamp (default: generate new one)")
    parser.add_argument("--start", type=int, default=1, help="Step number to start from (1-based)")
    parser.add_argument("--skip", nargs="*", type=int, default=[], help="Step numbers to skip")
    parser.add_argument("--use_config_lambda", action="store_true", help="Use lambda configuration settings if available")
    args_cli = parser.parse_args()


    exp_dir = Path(__file__).resolve().parent.parent / "experiments" / args_cli.experiment_name
    os.makedirs(exp_dir, exist_ok=True) 
    print(f"Experiment Directory: {exp_dir}")
    
    # Get timestamp (new or provided)
    timestamp = args_cli.timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # clusteirng folers results 
    base_results_dir = exp_dir/ "results" / f"run_{timestamp}"
    clustering_folders = [
        base_results_dir / "feature_extraction_AE",
        base_results_dir / "feature_extraction_AE_age",
        base_results_dir / "PCA",
        base_results_dir / "summary_statistics"
        # Add more folders as needed
    ]

    print(f"Using timestamp: {timestamp}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Pipeline starting from step {args_cli.start}, skipping: {args_cli.skip}")

    # Define pipeline stages with their run functions
    stages = [

        # 2. Base Autoencoder Training
        {
            "name": "Autoencoder Training",
            "func": run_training,
            "kwargs": {"timestamp": timestamp, "model_selection_timestamp": timestamp, "exp_dir": exp_dir}
        },

        # 4. Age-Aware Fine Tuning
        {
            "name": "Age-Aware AE Fine Tuning",
            "func": run_fine_tuning,
            "kwargs": {
                "timestamp": timestamp,
                "model_selection_timestamp": timestamp,
                "models_timestamp": timestamp,
                "use_config_lambda": args_cli.use_config_lambda,
                "exp_dir": exp_dir
            }
        },
        # 5. PCA Benchmark
        {
            "name": "PCA Benchmark",
            "func": run_pca_benchmark,
            "kwargs": {"timestamp": timestamp, "exp_dir": exp_dir}
        },
        # 6. Summary Statistics Benchmark
        {
            "name": "Summary Statistics Benchmark",
            "func": run_summary_statistics,
            "kwargs": {"timestamp": timestamp, "exp_dir": exp_dir}
        },
    ]

    # Execute pipeline
    for i, stage in enumerate(stages, start=1):
        if i < args_cli.start:
            print(f"Skipping step {i}: {stage['name']} (before --start)")
            continue
        if i in args_cli.skip:
            print(f"Skipping step {i}: {stage['name']} (in --skip)")
            continue

        try:
            print(f"\n{'='*70}")
            print(f"Step {i}/{len(stages)}: {stage['name']}")
            print(f"{'='*70}\n")
            
            stage["func"](**stage["kwargs"])
            
            print(f"\n✓ Step {i} completed successfully\n")
        except Exception as e:
            print(f"\n✗ Step {i} failed with error:")
            print(f"  {type(e).__name__}: {e}\n")
            raise  # Re-raise to stop pipeline on error

    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
