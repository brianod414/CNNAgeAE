#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import datetime
import pickle
import numpy as np
from sklearn.decomposition import PCA
import project1_a.scripts.utils as utils


def create_matrix_from_dict(patient_segments_dict, is_training_data: bool = False):
    # account for test data
    if is_training_data:
        patient_segments_dict = utils.flatten_training_windows(patient_segments_dict)

    matrix = np.vstack([
        df[["HRT", "BPm"]].values.T.flatten()
        for df in patient_segments_dict.values()
    ])
    print("Matrix shape:", matrix.shape)
    return matrix


def center(X_train: np.ndarray, X_test: np.ndarray | None = None):
    X_train_centered = X_train - X_train.mean(axis=0, keepdims=True)
    if X_test is not None:
        X_test_centered = X_test - X_train.mean(axis=0, keepdims=True)
        return X_train_centered, X_test_centered
    return X_train_centered


def prepare_PCA_matrices(patient_segments_dict, is_training_data: bool = False):
    if is_training_data:
        training_matrix = create_matrix_from_dict(patient_segments_dict, is_training_data=True)
        return training_matrix
    else:
        start_matrix = create_matrix_from_dict(patient_segments_dict["start"], is_training_data=False)
        end_matrix = create_matrix_from_dict(patient_segments_dict["end"], is_training_data=False)
        return start_matrix, end_matrix


def run_pca_benchmark(
    *,
    timestamp: str | None = None,
    config_path: str = "stages/benchmarks_config.yaml",
    exp_dir: Path
) -> Path:
    """
    Fit PCA on training data (HRT/BPm flattened) and transform test start/end, plus training start/end.

    Returns
    -------
    results_dir
    """
    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = "PCA"

    print(f"Starting PCA benchmark with config={config_path}, timestamp={timestamp}")

    results_dir, logs_dir = utils.setup_output_directories(
        exp_dir, timestamp, script=script_name, running_multi_iterations=False
    )
    utils.setup_logging(logs_dir, script_name, timestamp)

    # Load configuration
    config = utils.load_config(str(exp_dir.parent.parent / "configs" / config_path))
    data_config = utils.load_config(str(exp_dir.parent.parent / "configs" / "filenames.yaml"))

    # Write experiment parameters
    utils.write_experiment_parameters(
        config,
        results_dir,
        timestamp,
        args=None,
        experiment_name="PCA",
        description="Fit PCA to the test data, start and end segments separately. n_components chosen by variance threshold.",
    )

    # Load data
    train_filename = data_config["PCA"]["training_filename"]
    best_segments_train = utils.load_training_data(exp_dir, train_filename)

    test_filename = data_config["PCA"]["test_filename"]
    best_segments_test, selected_patient_ids = utils.load_test_data(exp_dir, test_filename)

    # Prepare matrices
    train_matrix = prepare_PCA_matrices(best_segments_train, is_training_data=True)
    start_matrix, end_matrix = prepare_PCA_matrices(best_segments_test)

    _, start_matrix_c = center(train_matrix, start_matrix)
    train_matrix_c, end_matrix_c = center(train_matrix, end_matrix)
    print(f"start test matrix: {start_matrix_c.shape}, end test matrix: {end_matrix_c.shape}")

    max_n_components = config["PCA"]["max_n_components"]
    variance_threshold = config["PCA"]["variance_threshold"]

    # Fit PCA to centered train
    pca_full = PCA(n_components=max_n_components)
    pca_full.fit(train_matrix_c)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    print("Cumulative explained variance:", cumulative_variance)

    n_components_opt = int(np.argmax(cumulative_variance >= variance_threshold) + 1)
    if n_components_opt == 1:
        n_components_opt = int(max_n_components)

    print(
        f"Optimal n_components: {n_components_opt} "
        f"(explains {cumulative_variance[n_components_opt-1]:.2%} variance)"
    )

    pca = PCA(n_components=n_components_opt)
    pca.fit(train_matrix_c)

    # Transform test
    pca_start = pca.transform(start_matrix_c)
    pca_end = pca.transform(end_matrix_c)

    PCA_results = {"PCA": {"start": pca_start, "end": pca_end}}

    # Also compute PCA features for training start/end windows (for cluster init etc.)
    lookup_period = config["data"]["training_start_end_lookup_period"]
    start_lookup = f"in_first_{lookup_period}h"
    end_lookup = f"in_last_{lookup_period}h"

    start_indices, end_indices = utils.create_indices_of_train_start_end_segments(
        best_segments_train, start_lookup, end_lookup
    )
    training_start_matrix = train_matrix[start_indices]
    training_end_matrix = train_matrix[end_indices]
    print(f"start training matrix: {training_start_matrix.shape}, end training matrix: {training_end_matrix.shape}")

    _, training_start_matrix_c = center(train_matrix, training_start_matrix)
    _, training_end_matrix_c = center(train_matrix, training_end_matrix)

    pca_start_features = pca.transform(training_start_matrix_c)
    pca_end_features = pca.transform(training_end_matrix_c)

    PCA_training_results = {"PCA": {"start": pca_start_features, "end": pca_end_features}}

    # Save results
    pca_results_path = results_dir / "PCA_results.pkl"
    with open(pca_results_path, "wb") as f:
        pickle.dump(PCA_results, f)

    pca_training_results_path = results_dir / "PCA_training_start_end_results.pkl"
    with open(pca_training_results_path, "wb") as f:
        pickle.dump(PCA_training_results, f)

    # Optional: save PCA model itself for reuse
    pca_model_path = results_dir / "pca_model.pkl"
    with open(pca_model_path, "wb") as f:
        pickle.dump(
            {
                "pca": pca,
                "n_components": n_components_opt,
                "train_mean": train_matrix.mean(axis=0),  # mean used for centering
            },
            f,
        )

    print("\nPCA benchmark completed successfully!")
    print(f"Results saved in: {pca_results_path}")
    print(f"Training start/end PCA saved in: {pca_training_results_path}")
    print(f"PCA model saved in: {pca_model_path}")

    return results_dir
