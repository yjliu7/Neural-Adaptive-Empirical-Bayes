import torch
import torch.nn as nn


class GaussianVariational(nn.Module):
    def __init__(self, mu, rho):
        """
        Gaussian Variational Weight Sampler
        Parameters:
            mu: tensor
                used to shift the samples drawn from a unit Gaussian
            rho: tensor
                used to generate the point-wise parameterization of the standard deviation
                and thus scale the samples drawn a unit Gaussian
        """
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)

        self.z = None
        self.sigma = None

        self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        """
        draw a sample from the posterior distribution using:
            z = mu + log(1 + exp(rho)) * epsilon
            where epsilon ~ N(0, 1)
        Returns:
            tensor: the sample from the posterior distribution
        """
        device = self.mu.device
        epsilon = self.normal.sample(self.mu.size()).to(device)
        self.sigma = torch.log(1 + torch.exp(self.rho)).to(device)
        self.z = self.mu + self.sigma * epsilon
        return self.z

    def kld(self):
        """
        use the sample from the posterior distribution to calculate the KL Divergence between
        the standard normal prior and posterior
        Returns:
            tensor: calculated KL Divergence
        """
        if self.z is None:
            raise ValueError('self.w must have a value.')
        kld = -0.5 * torch.sum(1 + 2 * torch.log(self.sigma) - self.mu.pow(2) - self.sigma.pow(2))
        return kld
