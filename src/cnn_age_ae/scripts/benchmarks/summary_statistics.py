#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import datetime
import pickle
import numpy as np

import project1_a.scripts.utils as utils

def create_HRT_BPm_matrices(patient_segments_dict, is_training_data: bool = False, no_standardisation: bool = False):
    """
    Create separate HRT and BPm matrices.

    Returns
    -------
    hrt_matrix : np.ndarray (n_patients, n_timepoints)
    bpm_matrix : np.ndarray (n_patients, n_timepoints)
    patient_ids : list
        IDs in row order
    """
    if is_training_data:
        patient_segments_dict = utils.flatten_training_windows(patient_segments_dict)

    patient_ids = list(patient_segments_dict.keys())

    hrt_matrix = np.vstack([df["HRT"].values for df in patient_segments_dict.values()])
    bpm_matrix = np.vstack([df["BPm"].values for df in patient_segments_dict.values()])

    if not no_standardisation:
        hrt_mean, hrt_std = hrt_matrix.mean(), hrt_matrix.std()
        bpm_mean, bpm_std = bpm_matrix.mean(), bpm_matrix.std()

        hrt_matrix = (hrt_matrix - hrt_mean) / (hrt_std if hrt_std > 0 else 1)
        bpm_matrix = (bpm_matrix - bpm_mean) / (bpm_std if bpm_std > 0 else 1)

    return hrt_matrix, bpm_matrix, patient_ids


def get_summary_stats_array(matrix: np.ndarray) -> np.ndarray:
    """
    Compute mean, min, max per patient and return as array.

    Returns
    -------
    stats_array : np.ndarray
        shape (n_patients, 3) = [mean, min, max]
    """
    means = matrix.mean(axis=1)
    mins = matrix.min(axis=1)
    maxs = matrix.max(axis=1)
    return np.vstack([means, mins, maxs]).T


def run_summary_statistics(
    *,
    timestamp: str | None = None,
    config_path: str = "stages/benchmarks_config.yaml",
    skip_training_data: bool = False,
    skip_test_data: bool = False,
    no_standardisation: bool = False,
    exp_dir: Path
) -> Path:
    """
    Extract summary statistics (mean/min/max) for HRT and BPm for:
      - training start/end (optional)
      - test start/end (optional)

    Returns
    -------
    results_dir (Path)
    """
    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = "summary_statistics"

    print(f"Starting summary statistics with config={config_path}, timestamp={timestamp}")

    results_dir, logs_dir = utils.setup_output_directories(
        exp_dir, timestamp, script=script_name, running_multi_iterations=False
    )
    utils.setup_logging(logs_dir, script_name, timestamp)

    # Load configs
    config = utils.load_config(str(exp_dir.parent.parent / "configs" / config_path))
    data_config = utils.load_config(str(exp_dir.parent.parent / "configs" / "filenames.yaml"))

    # Write experiment parameters
    utils.write_experiment_parameters(
        config,
        results_dir,
        timestamp,
        args=None,
        experiment_name="Summary Statistics",
        description="Extract summary statistics (mean, min, max) from start and end segments of HRT and BPm.",
    )

    # -------------------------------------------------------------------------- Training
    if not skip_training_data:
        train_filename = data_config["summary_stats"]["training_filename"]
        best_segments_train = utils.load_training_data(exp_dir, train_filename)

        HRT_matrix, BPm_matrix, ids = create_HRT_BPm_matrices(
            best_segments_train,
            is_training_data=True,
            no_standardisation=no_standardisation,
        )

        lookup_period = config["data"]["training_start_end_lookup_period"]
        start_lookup = f"in_first_{lookup_period}h"
        end_lookup = f"in_last_{lookup_period}h"

        start_indices, end_indices = utils.create_indices_of_train_start_end_segments(
            best_segments_train, start_lookup, end_lookup
        )

        # start
        HRT_matrix_start = HRT_matrix[start_indices]
        BPm_matrix_start = BPm_matrix[start_indices]
        start_stats_array = np.hstack([get_summary_stats_array(HRT_matrix_start), get_summary_stats_array(BPm_matrix_start)])

        # end
        HRT_matrix_end = HRT_matrix[end_indices]
        BPm_matrix_end = BPm_matrix[end_indices]
        end_stats_array = np.hstack([get_summary_stats_array(HRT_matrix_end), get_summary_stats_array(BPm_matrix_end)])

        training_results = {"summary_stats": {"start": start_stats_array, "end": end_stats_array}}

        results_path = results_dir / "summary_stats_training_start_end_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(training_results, f)
        print(f"Training start/end summary stats saved to: {results_path}")

    # -------------------------------------------------------------------------- Test
    if not skip_test_data:
        test_filename = data_config["summary_stats"]["test_filename"]
        best_segments_test, selected_patient_ids = utils.load_test_data(exp_dir, test_filename)

        # test start
        HRT_matrix, BPm_matrix, patient_ids = create_HRT_BPm_matrices(
            best_segments_test["start"],
            no_standardisation=no_standardisation,
        )
        start_stats_array = np.hstack([get_summary_stats_array(HRT_matrix), get_summary_stats_array(BPm_matrix)])

        # test end
        HRT_matrix, BPm_matrix, patient_ids = create_HRT_BPm_matrices(
            best_segments_test["end"],
            no_standardisation=no_standardisation,
        )
        end_stats_array = np.hstack([get_summary_stats_array(HRT_matrix), get_summary_stats_array(BPm_matrix)])

        test_results = {"summary_stats": {"start": start_stats_array, "end": end_stats_array}}

        results_path = results_dir / "summary_stats_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(test_results, f)
        print(f"Test summary stats saved to: {results_path}")

    print(f"Log file dir: {results_dir}")
    return results_dir
