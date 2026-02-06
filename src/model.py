import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder: Forcing the model to learn a very compact 4D representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 4), 
            nn.ReLU()
        )
        
        # Decoder: Reconstructing the features from 4D space
        self.decoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))