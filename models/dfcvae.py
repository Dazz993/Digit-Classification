'''
DFCVAE implemented with PyTorch
Refer to:
(1) https://arxiv.org/abs/1312.6114
(2) https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
(3) https://github.com/AntixK/PyTorch-VAE/blob/master/models/dfcvae.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn

class DFCVAE(nn.Module):
    def __init__(self, inplanes=3, input_size=32, latent_dim=128, hidden_dims=(32, 64, 128, 256, 512), alpha=1, beta=0.5):
        super(DFCVAE, self).__init__()
        # process arguments
        self.alpha = alpha
        self.beta = beta
        self.inplanes = inplanes
        self.lanten_dim = latent_dim
        hidden_dims = list(hidden_dims)

        # define encoder
        encoder_layers = []
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, hidden_dim, kernel_size=(3, 3), stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            inplanes = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.feature_map_size = feature_map_size = (input_size // (2 ** len(hidden_dims))) ** 2
        self.linear_mu = nn.Linear(inplanes * feature_map_size, latent_dim)
        self.linear_log_var = nn.Linear(inplanes * feature_map_size, latent_dim)

        # define decoder
        self.linear_decoder = nn.Linear(latent_dim, inplanes * feature_map_size)
        hidden_dims.reverse()
        decoder_layers = []
        for hidden_dim in hidden_dims[1:]:
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(inplanes, hidden_dim, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(inplace=True)
                )
            )
            inplanes = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(inplanes, inplanes, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(inplanes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inplanes, self.inplanes, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

        self.feature_network = vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()

    def encode(self, x):
        latent_tensor = self.encoder(x).flatten(1)
        mu = self.linear_mu(latent_tensor)
        log_var = self.linear_log_var(latent_tensor)
        return [mu, log_var]

    def decode(self, z):
        out = self.linear_decoder(z)
        batch, nelement = out.shape
        out = out.reshape(batch, -1, self.feature_map_size, self.feature_map_size)
        out = self.final_layer(self.decoder(out))
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def extract_features(self, x, feature_layers=None):
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = x
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if (key in feature_layers):
                features.append(result)

        return features

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructions = self.decode(z)

        recons_features = self.extract_features(reconstructions)
        input_features = self.extract_features(x)

        return [reconstructions, x, recons_features, input_features, mu, log_var]

    def loss(self, reconstuction, x, recons_features, input_features, mu, log_var, kl_weight=1):
        reconstruction_loss = F.mse_loss(reconstuction, x)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)

        loss = self.beta * (reconstruction_loss + feature_loss) + self.alpha * kl_weight * kl_loss

        return loss, reconstruction_loss, kl_loss

    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.lanten_dim).to(device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]

if __name__ == '__main__':
    vae = DFCVAE()
    input = torch.rand(128, 3, 32, 32)
    output = vae(input)