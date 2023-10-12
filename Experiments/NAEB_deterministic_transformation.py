import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim):
        """
        Deterministic Transformation Function G: W=G(Z)
        Parameters:
            latent_dim: int
                the dimension of the latent variable Z
            layer_sizes: list
                the hidden layers transforming Z to W
        """
        super().__init__()

        self.MLP = nn.Sequential()

        for layer_idx, (in_size, out_size) in enumerate(zip([latent_dim] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(layer_idx), module=nn.Linear(in_size, out_size))
            if layer_idx + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(layer_idx), module=nn.ReLU())

    def forward(self, z):
        """
        transform the latent variable Z to the weights of the neural network W (y=f_W(x))
        Parameters:
            z: tensor
                latent variable Z
        Returns:
            tensor: the weights W of the neural network
        """
        u = self.MLP(z)
        return u
