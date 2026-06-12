import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, Dataset

from project1_a.scripts.models.models import Conv1DAutoencoder
from project1_a.scripts.data_preparation.segment_selection_functions import interpolate_artefacts
from phd_general.loading_data_functions import load_demographic_data, load_age_based_HRT, load_age_based_BPm
from project1_a.scripts.models.early_stopping import EarlyStopping
from phd_general.data_helper_functions import assign_age_band, get_patient_ages

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------ DataSEts 
class SignalAgeDataset(Dataset):
    def __init__(self, X, y_age):
        """
        X: torch.Tensor of shape [N, C, T]
        y_age: torch.Tensor of shape [N] with `nan` for missing labels
        """
        self.X = X
        self.y_age = y_age

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_age[idx]


# ------------------------------------------------------ Data Formatting 
def create_tensor_from_list_segments(signals_dict, signal_names = ['HRT', 'BPm'], skip_tensor=False, return_id_tensor=False):
    ''' Function to create a tensor from the dictionary of best segments 
    
    Args:
        signals_dict (dict): dictionary of best window of signals for each patient
        Returns:
        combined_data (tensor): tensor of best segments for each patient
        '''

    # create list of segments for each signal
    print('Number of patients:', len(signals_dict.keys())) 
    all_signals = {signal: [] for signal in signal_names}
    n_segments = 0
    id_list = []

    for patient_id, window in signals_dict.items():
        n_segments += len(window.keys())

        # loop through windows for patients 
        for window_number, values in window.items():

            # interpolate artefacts in signals 
            clean_df = interpolate_artefacts(values['best_window'], signal_names)
            id_list.append(patient_id)

            for signal_name in signal_names:
                all_signals[signal_name].append(clean_df[signal_name].values)

    print('Total number of segments:', n_segments)

    # create a array, tensor for each signal then join tensors 
    signal_tensors = []
    for signal_name in signal_names:
        signal_array = np.array(all_signals[signal_name])
        print(f'{signal_name} shape:', signal_array.shape, '\nNans:', np.isnan(signal_array).sum())
        signal_tensor = torch.tensor(signal_array, dtype=torch.float32)
        signal_tensors.append(signal_tensor)

    # combine into a tensor - [n_patients, n_signals, n_time_steps]
    combined_data = torch.stack(signal_tensors, dim=1)

    # Print the shapes to verify
    print('Final tensor shape:', combined_data.shape)

    if return_id_tensor: 
        # Prepare patient IDs
        le = preprocessing.LabelEncoder()
        patient_ids = le.fit_transform(id_list)
        patient_ids = torch.tensor(patient_ids, dtype=torch.int64)
    else: 
        patient_ids = id_list 

    return combined_data, patient_ids 


def allocate_hold_out_data(training_data, validation_data_percentage=0.1):
    """Allocate hold-out data from training data"""
    # Calculate the number of samples to hold out
    n_patients = len(training_data)
    n_hout_patients = int(n_patients * validation_data_percentage)

    # Split the data into training and hold-out sets
    train_data = training_data[:-n_hout_patients]
    hold_out_data = training_data[-n_hout_patients:]

    return train_data, hold_out_data

# ------------------------------------------------------ Standardisation


def global_standardise_train_test_tensors(train_segments, test_segments = None):
    '''Function to standardise the train and test tensors using mean and std of train tensor
    
    The mean is one value per signal per.'''
    mean = train_segments.mean(dim = (0,2), keepdim=True)
    std = train_segments.std(dim = (0,2), keepdim=True)
    print(mean)
    print(std)

    # Normalize the data
    train_segments = (train_segments - mean) / std
    if test_segments is None:
        return train_segments
    else: 
        # Normalize the test data
        test_segments = (test_segments - mean) / std
        return train_segments, test_segments

def standardise_tensor_by_time(tensor):
    """ standardise tensor of signals independently per patient (per signal) along the time dimension. 
    n_patients x n_signals x 1 = dim of mean and std."""

    mean = tensor.mean(dim=(2), keepdim=True) # along the time dimension, 
    std = tensor.std(dim=(2), keepdim=True) # along the time dimension,

    standardised = (tensor - mean) / (std + 1e-8)  # Adding a small value to prevent division by zero
    return standardised


def per_patient_signal_standardisation(tensor_data, signal_names=['HRT', 'BPm']):
    """
    Standardize each patient's signal using their own mean and std over time.
    
    Parameters:
        tensor_data: torch.Tensor [patients x signals x time]
        signal_names: list of signal names in order (optional)
    
    Returns:
        Standardized tensor of same shape
    """
    # Create a copy to hold normalized data
    norm_tensor = tensor_data.clone()

    num_patients, num_signals, _ = tensor_data.shape

    for i in range(num_patients):
        for s in range(num_signals):
            signal = tensor_data[i, s, :]
            mean = signal.mean()
            std = signal.std()

            # Prevent divide-by-zero
            if std > 0:
                norm_tensor[i, s, :] = (signal - mean) / std
            else:
                norm_tensor[i, s, :] = signal - mean  # zero-centered if std is 0

    return norm_tensor


def age_based_range_normalisation(tensor_data, patient_ids, signal_names = ['HRT', 'BPm']):
    """
    Robust normalization using q05 and q95 from reference IQR tables for HRT and BPM.
    
    Parameters:
        tensor_data: torch.Tensor [patients x signals x time]
        patient_ids: list of patient IDs corresponding to tensor_data
        signal_names: list of signal names in order (e.g., ['HRT', 'BPM'])
        demographic_df: dataframe with 'patientID' and 'age_years'
        hrt_iqr_path: path to CSV with HRT age bands and IQRs
        bpm_iqr_path: path to CSV with BPM age bands and IQRs
        
    Returns:
        Normalized tensor of same shape
    """

    # Load IQR data and demographcis 
    demographic_df = load_demographic_data()
    hrt_iqr = load_age_based_HRT()
    bpm_iqr = load_age_based_BPm()

    # Build reference dictionary: {signal: {age_band: (q05, q95)}}
    ref_stats = {'HRT': {}, 'BPm': {}}

    for _, row in hrt_iqr.iterrows():

        age_band = (int(row['age_lower']), int(row['age_upper']))
        ref_stats['HRT'][age_band] = (row['q05'], row['q95'])
        print('HRT Row:', age_band)

    for _, row in bpm_iqr.iterrows():

        age_band = (int(row['age_lower']), int(row['age_upper']))
        ref_stats['BPm'][age_band] = (row['q05'], row['q95'])
        print('BPm Row:', age_band)

    print('Reference stats:', ref_stats)
    # Get sorted unique age band edges
    band_edges = sorted(set(hrt_iqr['age_lower']).union(hrt_iqr['age_upper']))

    print(ref_stats[signal_names[0]])
    print(ref_stats[signal_names[1]])
    # Create a tensor copy to normalize
    norm_tensor = tensor_data.clone()

    print('Tensor shape:', tensor_data.shape)

    # Map each patient to their age band and normalize their data per signal
    valid_patient_indices = []; valid_patient_ids = []; ages = []
    ages = get_patient_ages(patient_ids, demographic_df)
    for idx, pid in enumerate(patient_ids):

        # Get the patient's age
        age = ages[idx]
        if np.isnan(age):
            print(f"Patient {pid} has no age data, skipping normalization.")
            # remove age from the list
            ages[idx] = None
            print('removing')
            continue

        # Assign age band
        age_band = assign_age_band(age, band_edges)
        print('Age:', age, 'Band:', age_band)
        if age_band is None:
            print(f"Patient {pid} with age {age} not in any band")

        # If the patient has valid age data and age band, normalize their data
        valid_patient_indices.append(idx)
        valid_patient_ids.append(pid)

        
        for s_idx, signal in enumerate(signal_names):
            q05, q95 = ref_stats[signal][age_band]
            interval = q95 - q05
            midpoint = (q95 + q05) / 2
            norm_tensor[idx, s_idx, :] = (tensor_data[idx, s_idx, :] - midpoint) / interval

    # Create a new tensor with only the valid patients
    norm_tensor = norm_tensor[valid_patient_indices]
    patient_ids = valid_patient_ids

    # remove  None values from ages
    print('removing None values from ages')
    ages = [age for age in ages if age is not None]

    return norm_tensor, ages


def normalise_ages(training_ages, test_ages=None, method='minmax'):
    """Normalise ages to range [0, 1] based on min and max of training ages.

    Args:  
        training_ages (list): List of training ages. or tensor 
        test_ages (list, optional): List of test ages. If provided, will also be normalised.
        method (str): Normalisation method ('minmax' or 'zscore').  
    Returns:
        tuple: Normalised training ages and optionally normalised test ages.
    """

    # if training_ages is a tensor, convert to numpy
    if isinstance(training_ages, torch.Tensor):
        training_ages_np = training_ages.detach().cpu().numpy()
    else:
        training_ages_np = np.array(training_ages, dtype=np.float32)

    # if test_ages is a tensor, convert to numpy
    if isinstance(test_ages, torch.Tensor):
        test_ages_np = test_ages.detach().cpu().numpy()
    else:
        test_ages_np = np.array(test_ages, dtype=np.float32)

    if method == 'minmax':
        max_age = np.nanmax(training_ages_np)
        min_age = np.nanmin(training_ages_np)
        training_ages_np = (training_ages_np - min_age) / (max_age - min_age)

        if test_ages is not None:
            test_ages_np = (test_ages_np - min_age) / (max_age - min_age)

    elif method == 'zscore':
        mean_age = np.nanmean(training_ages_np)
        std_age = np.nanstd(training_ages_np)
        training_ages_np = (training_ages_np - mean_age) / std_age

        if test_ages is not None:
            test_ages_np = (test_ages_np - mean_age) / std_age

    return training_ages_np, test_ages_np if test_ages is not None else None


# ------------------------------------------------------ Data Preparation Functions
def prepare_data_for_model(model_name, input_data, device=None):
    """
    Prepare data based on model type requirements
    
    Args:
        model_name (str): 'Conv1D', 'Conv2D', or 'LSTM'
        train_data (torch.Tensor): Training data tensor
        val_data (torch.Tensor, optional): Validation data tensor
        device (torch.device, optional): Device to move data to
        
    Returns:
        tuple: (prepared_train_data, prepared_val_data) or just prepared_train_data if val_data is None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'Conv1D':
        # No additional preprocessing needed for Conv1D
        input_data = input_data.to(device)

    elif model_name == 'Conv2D':
        # Add channel dimension for Conv2D
        input_data = input_data.unsqueeze(1).to(device)  # Shape: [N, 1, C, T]
    elif model_name == 'LSTM':
        # Permute dimensions for LSTM (batch, time, features)
        input_data = input_data.permute(0, 2, 1).to(device)  # Shape: [N, T, C]

    else:
        raise ValueError(f"Unknown model type: {model_name}")


    return input_data



def prepare_training_val_tensors_and_standardise(best_segments_train, standardisation_method, percentage_hold_out, return_ids=False):
    """Create tensors from training data and apply standardisation
    
    The train test split is based on the ordering of patients and is deterministic"""
    # Create tensors
    combined_data, combined_data_patient_ids = create_tensor_from_list_segments(best_segments_train)
    print(f'Size of combined training data: {combined_data.shape}')
    
    training_data, validation_data = allocate_hold_out_data(combined_data, percentage_hold_out)
    training_ids, validation_data_ids = allocate_hold_out_data(combined_data_patient_ids, percentage_hold_out)

    
    if standardisation_method == 'global':
        # Standardise training data 
        training_data, validation_data = global_standardise_train_test_tensors(training_data, test_segments=validation_data)

    elif standardisation_method == 'age95':
        training_data, _ = age_based_range_normalisation(training_data, training_ids)
        validation_data, _ = age_based_range_normalisation(validation_data, validation_data_ids)

    elif standardisation_method == 'patient':
        training_data = per_patient_signal_standardisation(training_data)
        validation_data = per_patient_signal_standardisation(validation_data)
        
    elif standardisation_method == 'none':
        print("No standardisation applied")
        pass
    else:
        raise ValueError(f"Unknown standardisation method: {standardisation_method}")
    
    print(f"Applied {standardisation_method} standardisation to training data")

    if return_ids:
        return training_data, validation_data, training_ids, validation_data_ids
    else:       
        return training_data, validation_data




# ------------------------------------------------------ Model Training and evaluation 

def train_AE_batch_model(model, optimizer, criterion, train_loader, epochs, val_loader=None, early_stopping=None, loud=False):
    ''' Function to train the autoencoder model (reconstruction model)
    
    Args:
        model: the autoencoder model
        optimizer: the optimizer for training
        criterion: the loss function (MSE or MAE)
        train_loader: the training data loader
        epochs: number of epochs for training
        val_loader: the validation data loader
        early_stopping: EarlyStopping object for early stopping
    Return:
        losses: list of losses for each epoch
        '''
    # list to store losses 
    losses = {'train': [], 'val': []}

    # loop through epochs
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstruction, _, _ = model(batch)
            loss = criterion(reconstruction, batch)

            if loud:
                print(f"Batch size: {batch.shape[0]}, Signal shape: {batch.shape}, Reconstruction shape: {reconstruction.shape}")

            # training 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        losses['train'].append(avg_train_loss)

        # evaluate model on validation data
        if val_loader:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    reconstruction, _, _ = model(batch)
                    loss = criterion(reconstruction, batch)
                    val_loss += loss.item()*batch.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            losses['val'].append(avg_val_loss)
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')


        # early stopping
        stopped_epoch = epochs  # default if early stopping never triggers

        if val_loader and early_stopping:
            early_stopping(avg_val_loss, model, epoch=epoch)
            if early_stopping.early_stop:

                # replace model weights with best model weights
                early_stopping.load_best_model(model)
                stopped_epoch = early_stopping.get_best_epoch()
                print(f"Early stopping at epoch {stopped_epoch}")
                break

    return losses, stopped_epoch



def train_AE_age_batch_model(model, ae_optimizer, age_optimizer, ae_criterion, age_criterion, 
                             train_loader, val_loader=None, early_stopping=None, lambda_age=0,
                             lambda_separation=0, pseudo_age=False, epochs=100, device=torch.device('cpu')):
    """
    Train an autoencoder model with an auxiliary age regression task.

    Args:
        model: The autoencoder model with an age predictor.
        ae_optimizer: The optimizer for the autoencoder.
        age_optimizer: The optimizer for the age predictor.
        train_loader: DataLoader for training data.
        lambda_age: Weight for the age prediction loss.
        lambda_separation: Weight for the age separation loss.
        pseudo_age: Whether to use pseudo age labels.
        epochs: Number of training epochs.

    This function trains the model by iterating over the training data in batches.
    Reconstruct the signal, compute an age prediction from the latent representation 
    Compute the loss using age prediction and reconstruction. 

    Returns:
        results_dict: Dictionary containing losses and final age predictions/labels.

    """

    losses = {'train': {'total': [], 
                        'recon': [],
                        'age': [],
                        'supervised_age': [], 
                        'pseudo_age': [], 
                        'age_separation': []
                        }, 
              'val':{'total':[],
                     'recon': [], 
                     'age': [], 
                     'supervised_age': [], 
                     'pseudo_age': [], 
                     'age_separation': []
                    }
            }

    

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}****************************************")

        total_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        age_loss_epoch = 0.0
        supervised_age_loss_epoch = 0.0
        pseudo_age_loss_epoch = 0.0
        age_separation_loss_epoch = 0.0 
        epoch_age_predictions = []
        epoch_age_labels = []

        # iterate over batches 
        for x, y_age in train_loader:
            x = x.to(device)
            y_age = y_age.to(device)
            n_samples = x.size(0)
            # print(f"Batch size: {x.shape[0]}, Signal shape: {x.shape}, Age labels shape: {y_age.shape}")
            y_age = y_age.unsqueeze(1).float()

            # fit the model 
            ae_optimizer.zero_grad()
            age_optimizer.zero_grad()
            reconstruction, z, age_pred = model(x, age_label=y_age)

            if torch.isnan(reconstruction).any() or torch.isnan(z).any():
                print("NaN in reconstruction or latent vector. Skipping batch.")
                continue

            # compute the loss 
            loss, recon_loss, age_loss, age_separation_loss, (supervised_age_loss, pseudo_age_loss) = model.compute_loss(
                x, z, reconstruction,
                ae_criterion, age_criterion, 
                age_pred=age_pred,
                age_true=y_age,
                lambda_age=lambda_age,
                lambda_separation=lambda_separation,
                pseudo_age=pseudo_age
            )

            # Skip batch if loss is NaN
            if torch.isnan(loss).any():
                print("Skipping batch with NaN loss.***********************************")
                continue
        
            loss.backward()



            ae_optimizer.step()
            age_optimizer.step()

            # Track values for monitoring 
            # Mask NaN age labels
            valid_age_mask = ~torch.isnan(y_age)
            if valid_age_mask.any():  # Only append if there are valid ages
                valid_age_labels = y_age[valid_age_mask].squeeze().detach().cpu().numpy()
                valid_age_predictions = age_pred[valid_age_mask].squeeze().detach().cpu().numpy()
                
                # Ensure arrays are always 1D (convert scalars to 1D arrays)
                if valid_age_labels.ndim == 0:
                    valid_age_labels = np.array([valid_age_labels])
                if valid_age_predictions.ndim == 0:
                    valid_age_predictions = np.array([valid_age_predictions])
                    
                epoch_age_labels.append(valid_age_labels)
                epoch_age_predictions.append(valid_age_predictions)

            # Accumulate losses (FIXED: only accumulate once)
            total_loss_epoch += loss.item()*n_samples
            recon_loss_epoch += recon_loss.item()*n_samples

            # Only accumulate age losses if they exist (not None)
            if age_loss is not None:
                age_loss_epoch += age_loss.item()*n_samples
            if supervised_age_loss is not None:
                supervised_age_loss_epoch += supervised_age_loss.item()*n_samples
            if pseudo_age_loss is not None:
                pseudo_age_loss_epoch += pseudo_age_loss.item()*n_samples
            if age_separation_loss is not None:
                age_separation_loss_epoch += age_separation_loss.item()*n_samples

        # save average losses
        losses['train']['total'].append(total_loss_epoch/len(train_loader.dataset))
        losses['train']['recon'].append(recon_loss_epoch/len(train_loader.dataset))
        losses['train']['age'].append(age_loss_epoch/len(train_loader.dataset))
        losses['train']['supervised_age'].append(supervised_age_loss_epoch/len(train_loader.dataset))
        losses['train']['pseudo_age'].append(pseudo_age_loss_epoch/len(train_loader.dataset))
        losses['train']['age_separation'].append(age_separation_loss_epoch/len(train_loader.dataset))

        # evaluate model on validation data 
        if val_loader: 
            model.eval()
            val_total_loss_epoch = 0
            val_recon_loss_epoch = 0
            val_age_loss_epoch = 0
            val_supervised_age_loss_epoch = 0
            val_pseudo_age_loss_epoch = 0
            val_age_separation_loss_epoch = 0

            with torch.no_grad():
                for (x, y_age) in val_loader:
                    x = x.to(device)
                    y_age = y_age.to(device)
                    n_samples = x.size(0)
                    reconstruction, z, age_pred = model(x, age_label=y_age.unsqueeze(1).float())
                    loss, recon_loss, age_loss, age_separation_loss, (supervised_age_loss, pseudo_age_loss) = model.compute_loss(
                        x, z, reconstruction,
                        ae_criterion, age_criterion, 
                        age_pred=age_pred,
                        age_true=y_age,
                        lambda_age=lambda_age,
                        lambda_separation=lambda_separation,
                        pseudo_age=pseudo_age
                    )
                    val_total_loss_epoch += loss.item()*n_samples
                    val_recon_loss_epoch += recon_loss.item()*n_samples

                    # Only accumulate age losses if they exist (not None)
                    if age_loss is not None:
                        val_age_loss_epoch += age_loss.item()*n_samples
                    if supervised_age_loss is not None:
                        val_supervised_age_loss_epoch += supervised_age_loss.item()*n_samples
                    if pseudo_age_loss is not None:
                        val_pseudo_age_loss_epoch += pseudo_age_loss.item()*n_samples
                    if age_separation_loss is not None:
                        val_age_separation_loss_epoch += age_separation_loss.item()*n_samples

                # average val_loss
                avg_val_loss = val_total_loss_epoch/len(val_loader.dataset)
                losses['val']['total'].append(avg_val_loss)
                losses['val']['recon'].append(val_recon_loss_epoch/len(val_loader.dataset))
                losses['val']['age'].append(val_age_loss_epoch/len(val_loader.dataset))
                losses['val']['supervised_age'].append(val_supervised_age_loss_epoch/len(val_loader.dataset))
                losses['val']['pseudo_age'].append(val_pseudo_age_loss_epoch/len(val_loader.dataset))
                losses['val']['age_separation'].append(age_separation_loss_epoch/len(val_loader.dataset))
                print(f'Epoch {epoch+1}, Train Loss: {avg_val_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_val_loss:.4f}')


        # do early stopping 
        stopped_epoch = epoch
        if val_loader and early_stopping:
            early_stopping(avg_val_loss, model, epoch=epoch)
            if early_stopping.early_stop:
                # replace model weights with best weights 
                early_stopping.load_best_model(model)
                stopped_epoch = early_stopping.get_best_epoch()
                print(f"Early stopping triggered. Best epoch: {stopped_epoch}")
                break

    return losses, stopped_epoch


# ------------------------------------------------------------------------------------ Loading Models 
def load_best_model_params(model_selection_timestamp, model_name, model_selection_dir, is_age_model=False):
    import yaml

    if is_age_model:
        model_selection_path = model_selection_dir / 'results' / f'run_{model_selection_timestamp}' / "model_selection_AE_age" / f"best_params_{model_name}.yaml"

    else:
        model_selection_path = model_selection_dir / 'results' / f'run_{model_selection_timestamp}' / "model_selection" / f"best_params_{model_name}.yaml"


    print('Loading best model parameters from:', model_selection_path)
    with open(model_selection_path, 'r') as f:
        best_params = yaml.safe_load(f)

    print("loaded best_params wth keys:", best_params.keys())
    if is_age_model:
        return best_params['lambda_age']
    else: 
        return best_params['learning_rate'], best_params['num_layers'], best_params['encoding_dim']

