
# Imports
import yaml
import pickle
from pathlib import Path
import torch
import omegaconf

from phd_general import data_helper_functions
from project1_a.scripts.models import autoencoder_fitting
# ------------------------ General 
def setup_output_directories(exp_dir, timestamp, script, running_multi_iterations=False):
    """Create output directories for results and logs"""

    results_dir = exp_dir / "results" / f"run_{timestamp}" / script
    logs_dir = exp_dir / "logs"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir, logs_dir


def merge_configs(base_config, override_config):

    config = omegaconf.OmegaConf.merge(base_config, override_config)
    return config

def load_config(config_filepath=None, config_name=None):
    """Load configuration from YAML file
    Args: config filepath or config dir + name"""

    # config name and exp_dir provided 
    if config_name is not None:
        project_dir = Path(__file__).resolve().parent.parent.parent.parent
        config_path = project_dir / 'configs' / config_name
    # full path provided 
    elif config_filepath is not None:
        config_path = Path(config_filepath)
        print("FULL PATH PROVIDED:", config_path)
    else: 
        raise ValueError("Either config_filepath or config_name must be provided")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")

    return config

def save_config(config, config_name):
    """Save configuration to YAML file in configs directory"""
    project_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = project_dir / 'configs' / config_name
    # craete dir if it doesn't exist
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Configuration saved to: {config_path}")
    return config_path


# ------------------------ Training Functions 


def load_training_data(exp_dir, training_data_filename):
    """Load training data from pickle files"""
    # Load training data
    training_path = exp_dir.parent.parent / 'interim_data' / training_data_filename

    try: 
        with open(training_path, 'rb') as f:
            best_segments_train = pickle.load(f)
            print(f'Loaded {training_data_filename}')
    except FileNotFoundError:
        print(f'File {training_data_filename} not found at {training_path}')
        raise
    except Exception as e:
        print(f'An error occurred loading training data: {e}')
        raise
    
    return best_segments_train

def get_normalised_patient_ages(combined_data_patient_ids):
    """Add age information to training segments if available"""
    training_ages = data_helper_functions.get_patient_ages(combined_data_patient_ids)
    print(f'Number of patients with age data: {len(training_ages)}')

    training_ages, _ = autoencoder_fitting.normalise_ages(training_ages, test_ages=None, method='minmax')

    return training_ages







# -------------------------------------- Experiment Management Functions


def flatten_training_windows(dict):
    # flattend the nested dict of training windows to a single level dict
    flattened_data = {}
    for patient_id, windows in dict.items():
        for window_id, segments in windows.items():
            data = segments["best_window"]
            id = f"patient_{patient_id}_{window_id}"
            flattened_data[id] = data
    return flattened_data


def save_training_results(training_meta_data, results_dir, running_multi_iterations, run_index, filename_prefix="experiment7_02_model_training_meta_data"):
    """Save training meta data to pickle files
    
    Args:
        training_meta_data: Dictionary containing training results
        results_dir: Path to results directory
        running_multi_iterations: Whether running multiple iterations
        run_index: Index of current run
        filename_prefix: Prefix for output filename (default: experiment7_02_model_training_meta_data)
    
    Returns:
        Path to saved metadata file
    """
    def move_dict_tensors_to_cpu(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.cpu()
            elif isinstance(v, dict):
                d[k] = {ik: iv.cpu() if isinstance(iv, torch.Tensor) else iv for ik, iv in v.items()}
        return d

    for model_name in training_meta_data:
        if isinstance(training_meta_data[model_name], dict):
            training_meta_data[model_name] = move_dict_tensors_to_cpu(training_meta_data[model_name])
        elif isinstance(training_meta_data[model_name], torch.Tensor):
            training_meta_data[model_name] = training_meta_data[model_name].cpu()

    if running_multi_iterations:
        meta_data_filename = f"{filename_prefix}_run{run_index}.pkl"
    else:
        meta_data_filename = f"{filename_prefix}.pkl"

    meta_data_path = results_dir / meta_data_filename
    with open(meta_data_path, "wb") as f:
        pickle.dump(training_meta_data, f)
        print(f"Training meta data saved to {meta_data_path}")

    return meta_data_path


