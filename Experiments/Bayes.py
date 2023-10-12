import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from deterministic_transformation import Decoder
from gaussian_variational import GaussianVariational


def mini_batch_weight(batch_idx, num_batches):
    """
    calculate the mini-batch weight which decreases as the batch index increases
    Parameters:
        batch_idx: int
            current batch index
        num_batches: int
            total number of batches
    Returns:
        float: current mini-batch weight
    """
    return 2 ** (num_batches - batch_idx) / (2 ** num_batches - 1)


class BayesianNetwork(nn.Module):
    def __init__(self, latent_dim, hidden1_size, hidden2_size, x_dim, y_dim, decoder_layer_sizes):
        """
        Neural Adaptive Empirical Bayes: construct a two-hidden-layer MLP f_W for example
        Parameters:
            latent_dim: int
                the dimension of the latent variable Z
            hidden1_size: int
                the size of the first hidden layer
            hidden2_size: int
                the size of the second hidden layer
            x_dim: int
                the dimension of x
            y_dim: int
                the dimension of y
            decoder_layer_sizes: list
                the hidden layers transforming Z to W
        """
        super().__init__()
        z_mu = torch.empty(1, latent_dim).uniform_(-1, 1)
        z_rho = torch.empty(1, latent_dim).uniform_(-1, 1)

        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_posterior = GaussianVariational(z_mu, z_rho)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim)
        self.kl_divergence = 0.0

    def forward(self, x):
        """
        map data inputs x to their outputs y via a neural network f_W
        for a two-hidden-layer neural network f_W, the dimension of W:
        x_dim * hidden1_size + hidden1_size + hidden1_size * hidden2_size + hidden2_size + hidden2_size * y_dim + y_dim
        Parameters:
            x: tensor
                data inputs x
        Returns:
            tensor: outputs y
        """
        if x.dim() > 2:
            x = x.view(-1, 28 * 28)
        z = self.z_posterior.sample()
        u = self.decoder(z)
        w1 = u[:, 0:self.x_dim * self.hidden1_size]
        w1 = w1.view((self.x_dim, -1))
        b1 = u[:, self.x_dim * self.hidden1_size:(self.x_dim + 1) * self.hidden1_size]
        b1 = b1.view((1, -1))
        w2 = u[:, (self.x_dim + 1) * self.hidden1_size:(self.x_dim + 1 + self.hidden2_size) * self.hidden1_size]
        w2 = w2.view((self.hidden1_size, -1))
        b2 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size:
                  (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + self.hidden2_size]
        b2 = b2.view((1, -1))
        w3 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + self.hidden2_size:
                  (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + (1 + self.y_dim) * self.hidden2_size]
        w3 = w3.view(self.hidden2_size, -1)
        b3 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + (1 + self.y_dim) * self.hidden2_size:]
        b3 = b3.view((1, -1))

        z_kld = self.z_posterior.kld()
        self.kl_divergence = z_kld
        hidden1 = F.relu(torch.mm(x, w1) + b1)
        hidden2 = F.relu(torch.mm(hidden1, w2) + b2)
        output = torch.mm(hidden2, w3) + b3
        return output

    def elbo(self, inputs, targets, criterion, n_samples, w_complexity=1.0):
        """
        sample the ELBO loss for a given batch of data
        Parameters:
            inputs : tensor
                inputs x to the model
            targets : tensor
                target outputs y of the model
            criterion : any
                loss function used to calculate data-dependant loss
            n_samples : int
                number of Monte Carlo samples to use
            w_complexity : float
                complexity weight multiplier
        Returns:
            tensor: value of the ELBO loss for the given data.
        """
        loss = 0
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += self.kl_divergence * w_complexity
        return loss / n_samples

    def inference(self, x, inference_size):
        """
        generate the averaged predictions given a new input x
        Parameters:
            x: tensor
                new inputs
            inference_size: int
                the number of generated predictions
        Returns:
            tensor: predicted labels
        """
        if x.dim() > 2:
            x = x.view(-1, 28 * 28)
        device = x.device
        predict_y = torch.zeros((inference_size, x.shape[0], self.y_dim)).to(device)
        with torch.no_grad():
            for inf_idx in range(inference_size):
                z = self.z_posterior.sample()
                u = self.decoder(z)
                w1 = u[:, 0:self.x_dim * self.hidden1_size]
                w1 = w1.view((self.x_dim, -1))
                b1 = u[:, self.x_dim * self.hidden1_size:(self.x_dim + 1) * self.hidden1_size]
                b1 = b1.view((1, -1))
                w2 = u[:, (self.x_dim + 1) * self.hidden1_size:(self.x_dim + 1 + self.hidden2_size) * self.hidden1_size]
                w2 = w2.view((self.hidden1_size, -1))
                b2 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size:
                          (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + self.hidden2_size]
                b2 = b2.view((1, -1))
                w3 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size + self.hidden2_size:
                          (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size +
                          (1 + self.y_dim) * self.hidden2_size]
                w3 = w3.view(self.hidden2_size, -1)
                b3 = u[:, (self.x_dim + 1 + self.hidden2_size) * self.hidden1_size +
                          (1 + self.y_dim) * self.hidden2_size:]
                b3 = b3.view((1, -1))

                hidden1 = F.relu(torch.mm(x, w1) + b1)
                hidden2 = F.relu(torch.mm(hidden1, w2) + b2)
                output = torch.mm(hidden2, w3) + b3
                predict_y[inf_idx, :, :] = output
        return predict_y
