### ---------------------------
# Script Name: models.py
# Scrpit Description: This script contains class definitions for models and clustering methods 
## ---------------------------

# standard libraries 
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# clustering libraries 
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


# --------------------- Autoencoder classes ---------------------

# create a base class for autoencoders
class BaseAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.encoding_dim = encoding_dim

    def forward(self, x):
        raise NotImplementedError("BaseAutoencoder is abstract. Implement in subclasses.")
    

class AgeRegressionMixin:
    def init_age_head(self, encoding_dim):
        ''' Predict age from encoding dimension '''

        # Using a pure linear regressor
        # self.age_predictor = nn.Linear(encoding_dim, 1)

        # Using a simple non-linear regressor with one hidden layer
        self.age_predictor = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
        )


    def predict_age(self, z):
        ''' Predict age from the latent representation z.
        Args:
            z (Tensor): Latent representation from the autoencoder.
        Returns:
            age_pred (Tensor): Predicted ages.
        '''
        return self.age_predictor(z)


class AutoencoderLossMixin(nn.Module):

    def __init__(self):
        super().__init__()
        self._latest_z_dist_matrix = None
        self._latest_age_diff_matrix = None


    def compute_loss(self, x, z, recon, ae_criterion, age_criterion, age_pred=None, age_true=None, lambda_age=0, lambda_separation=0, pseudo_age=False):
        ''' Compute the loss for the autoencoder.
        Args:
            x (Tensor): Original input data.
            recon (Tensor): Reconstructed data from the autoencoder.
            age_pred (Tensor, optional): Predicted ages from the age head.
            age_true (Tensor, optional): True ages for computing age loss.
            lambda_age (float, optional): Weight for the age loss term. Default is 1.0.
            return_matrices (bool, optional): If True, returns distance matrices along with losses.
            
        Returns:
            total_loss (Tensor): Total loss combining reconstruction and age loss.
            recon_loss (Tensor): Reconstruction loss.
            age_loss (Tensor, optional): Age loss if applicable, otherwise None.
            age_separation_loss (Tensor): Age separation loss.
            (supervised_age_loss, pseudo_age_loss) (tuple): Individual age loss components.
            z_dist_matrix (Tensor, optional): Distance matrix of latent representations (if return_matrices=True).
            age_diff_matrix (Tensor, optional): Squared age difference matrix (if return_matrices=True).
        '''

        # Reconstruction loss
        recon_loss = ae_criterion(recon, x)

        if age_pred is None or age_true is None:
            # If no age prediction or true age is provided, return only reconstruction loss
            print("No age prediction or true age provided. Returning only reconstruction loss.")
            return recon_loss, recon_loss, None, torch.tensor(0.0, device=x.device), (torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device))
        
        else:
            # Squeeze age tensors 
            age_true = age_true.squeeze().to(age_pred.device) # move true values to pred_device 
            age_pred = age_pred.squeeze()

            # Masks for missing age labels
            has_age = ~torch.isnan(age_true)
            missing_age = torch.isnan(age_true)
            # Ensure masks are on the same device as age_true/age_pred
            has_age = has_age.to(age_true.device)
            missing_age = missing_age.to(age_true.device)

            # Initialize losses
            supervised_age_loss = torch.tensor(0.0, device=x.device)
            pseudo_age_loss = torch.tensor(0.0, device=x.device)

            # Supervised loss (on valid age labels)
            if has_age.any():
                # print(f'Computing supervised age loss for {has_age.sum()} samples.')
                supervised_age_loss = age_criterion(age_pred[has_age], age_true[has_age])

            # Pseudo-label loss (on missing labels)
            if pseudo_age and missing_age.any():
                # print(f'Computing pseudo age loss for {missing_age.sum()} samples.')
                pseudo_targets = age_pred[missing_age].detach()  # stop gradient here
                # add tiny noise to avoid 0 loss
                pseudo_targets += torch.randn_like(pseudo_targets) * 0.1
                pseudo_age_loss = age_criterion(age_pred[missing_age], pseudo_targets)
                # print("Pseudo Age: \n Targets = ", pseudo_targets, "\n prediction = ", age_pred[missing_age])
            if not pseudo_age and missing_age.any():
                print(f'No pseudo age loss computed as pseudo_age is set to False for {missing_age.sum()} samples.')
                 

            # Total age loss
            age_loss = supervised_age_loss + pseudo_age_loss

            # --- Age Separation Loss 
            if lambda_separation == 0:
                age_separation_loss = torch.tensor(0.0, device=x.device)
            else:
                # Compute age separation loss only if there are valid ages
                # print(f'Computing age separation loss for {has_age.sum()} samples.')
                age_separation_loss = torch.tensor(0.0, device=x.device)

                if has_age.sum() >=2:
                    # get latent variable and age for a non-missing age 
                    z_known = z[has_age]
                    age_known = age_true[has_age].unsqueeze(1)
                    # compute the age separation loss
                    age_separation_loss = self._compute_age_separation_loss(z_known, age_known)

            # compute total loss 
            total_loss = (
                recon_loss 
                + lambda_age * age_loss
                + lambda_separation * age_separation_loss
            )

            return total_loss, recon_loss, age_loss, age_separation_loss, (supervised_age_loss, pseudo_age_loss)
        

    def _compute_age_separation_loss(self, z, ages):
        '''Encourages embedding distances to reflect age differences'''

        n = z.size(0)
        print(f'Latent shape: {z.shape}, Age shape: {ages.shape}')

        # get squared distances between ages and z values 
        age_diff_matrix = (ages.unsqueeze(0) - ages.unsqueeze(1)) ** 2
        z_dist_matrix = torch.cdist(z, z, p=2) ** 2

        # Store for later access
        self._latest_z_dist_matrix = z_dist_matrix.detach().cpu()
        self._latest_age_diff_matrix = age_diff_matrix.detach().cpu()


        loss = F.smooth_l1_loss(z_dist_matrix, age_diff_matrix)


        return loss


    def get_latest_distance_matrices(self):
        """
        Returns:
            z_dist_matrix (Tensor or None): Pairwise latent distances.
            age_diff_matrix (Tensor or None): Pairwise age difference squared.
        """
        return self._latest_z_dist_matrix, self._latest_age_diff_matrix


# 1D CNN
class Conv1DAutoencoder(BaseAutoencoder, AgeRegressionMixin, AutoencoderLossMixin):
    def __init__(self, input_channels=2, encoding_dim=2, num_layers=1, use_age_regression=False):
        super().__init__(input_size=input_channels, encoding_dim=encoding_dim)
        self.use_age_regression = use_age_regression

        # Only initialize the age head if needed
        if self.use_age_regression:
            AgeRegressionMixin.init_age_head(self, encoding_dim=encoding_dim)

        # Encoder 
        encoder_layers = []
        in_channels = input_channels 

        # itertate through the number of layers
        for layer in range(num_layers):
            # convolutional layer 
            encoder_layers.append(nn.Conv1d(in_channels, encoding_dim, kernel_size=3, stride=1, padding=0))
            # leaky rely    
            encoder_layers.append(nn.LeakyReLU(True))
            # average pooling across time dimension 
            encoder_layers.append(nn.AvgPool1d(kernel_size=3, stride=1))
            # input channels of next layer is the size of encoding 
            in_channels = encoding_dim  
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        out_channels = encoding_dim

        # iterate through the number of layers        
        for layer in range(num_layers):
            # reverse the AvgPool1d
            decoder_layers.append(nn.Upsample(size=360, mode='linear'))
            # leaky relu
            decoder_layers.append(nn.LeakyReLU(True))
            
            # Only set `out_channels = input_channels` on the last layer
            if layer == num_layers - 1:
                out_channels = input_channels
            # convolutional layer with padding to maintain the size of the input
            decoder_layers.append(nn.ConvTranspose1d(encoding_dim, out_channels, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)




    def forward(self, x, age_label=None, loud=False):

        # encode the input
        encoded = self.encoder(x)
        # average pooling to reduce the time dimension to 1
        z = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
        # decode the encoded representation
        decoded = self.decoder(encoded)
        
        if loud:
            print(f'z shape: {z.shape}, encoded shape: {encoded.shape}, decoded shape: {decoded.shape}')

        if self.use_age_regression:
            # predict age from the latent representation (1 dim)
            age_pred = self.predict_age(z)
            return decoded, z, age_pred
        
        return decoded, z, None


# --------------------- Clustering classes ---------------------

class KMeansClustering():
    def __init__(self, n_clusters=3, n_init=10, random_state=None, init_centroids="k-means++"):

        # single initialisation when centroids provided 
        if init_centroids is not None: 
            n_init = 1
        # update centroids if None to "k-means++"
        elif init_centroids is None:
            init_centroids = "k-means++"
        self.n_init = n_init
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init_centroids
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init, init=self.init)

    def fit(self, data):
        self.model.fit(data)
        return self.model
    
    def predict(self, data):
        return self.model.predict(data)
    
    def get_cluster_centers(self):
        return self.model.cluster_centers_
    
    def get_labels(self):
        return self.model.labels_
    
    def get_inertia(self):
        return self.model.inertia_
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model
    


class HierarchicalClusering():
    def __init__(self, n_clusters = 2, linkage = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
    
    def fit(self, data):
        self.model.fit(data)
        return self.model
    
    def get_labels(self):
        return self.model.labels_
    

