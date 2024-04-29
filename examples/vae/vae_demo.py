from __future__ import print_function

import argparse
import os
from typing import Any

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train (default: 10)',
    )
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )
    args = parser.parse_args()
    return args


class VAE(nn.Module):
    """A Variational Autoencoder (VAE) model for MNIST dataset.

    Args:
        None

    Attributes:
        fc1 (nn.Linear): A fully connected layer with 400 neurons.
        fc21 (nn.Linear): A fully connected layer with 20 neurons.
        fc22 (nn.Linear): A fully connected layer with 20 neurons.
        fc3 (nn.Linear): A fully connected layer with 400 neurons.
        fc4 (nn.Linear): A fully connected layer with 784 neurons.

    Methods:
        __init__(self): Initializes the VAE model.
        encode(self, x): Encodes the input tensor x and returns the mean of the latent distribution (mu) and the logarithm of the variance of the latent distribution (logvar).
        reparameterize(self, mu, logvar): Reparameterizes the latent distribution using the given mean (mu) and logarithm of the variance (logvar).
        decode(self, z): Decodes the latent tensor z and returns the reconstructed image.
        forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Any, Any]: Encodes the input tensor x and returns the decoded tensor, the mean of the latent distribution (mu), and the logarithm of the variance of the latent distribution (logvar).
    """

    def __init__(self):
        """Initializes the VAE model.

        Args:
            None

        Attributes:
            fc1 (nn.Linear): A fully connected layer with 400 neurons.
            fc21 (nn.Linear): A fully connected layer with 20 neurons.
            fc22 (nn.Linear): A fully connected layer with 20 neurons.
            fc3 (nn.Linear): A fully connected layer with 400 neurons.
            fc4 (nn.Linear): A fully connected layer with 784 neurons.
        """
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x: torch.Tensor):
        """Encodes the input tensor x and returns the mean of the latent
        distribution (mu) and the logarithm of the variance of the latent
        distribution (logvar).

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 784), where 784 is the number of pixels in a MNIST image.

        Returns:
            tuple[torch.Tensor, Any]: A tuple containing the decoded tensor (mu), and the logarithm of the variance of the latent distribution (logvar).
        """
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterizes the latent distribution using the given mean (mu)
        and logarithm of the variance (logvar).

        Args:
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The logarithm of the variance of the latent distribution.

        Returns:
            torch.Tensor: A tensor representing the reparameterized latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """Decodes the latent tensor z and returns the reconstructed image.

        Args:
            z (torch.Tensor): The latent tensor of shape (batch_size, 20).

        Returns:
            torch.Tensor: A tensor representing the reconstructed image.
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Any, Any]:
        """Encodes the input tensor x and returns the decoded tensor, the mean
        of the latent distribution (mu), and the logarithm of the variance of
        the latent distribution (logvar).

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 784), where 784 is the number of pixels in a MNIST image.

        Returns:
            tuple[torch.Tensor, Any, Any]: A tuple containing the decoded tensor (recon_x), the mean of the latent distribution (mu), and the logarithm of the variance of the latent distribution (logvar).
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, optimizer, loader, epoch, args) -> None:
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(loader.dataset),
                100.0 * batch_idx / len(loader),
                loss.item() / len(data),
            ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader.dataset)))


def test(model, loader, epoch, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(args.device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n],
                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]
                ])
                file_name = os.path.join(
                    args.work_dir, 'reconstruction_' + str(epoch) + '.png')
                save_image(
                    comparison.cpu(),
                    file_name,
                    nrow=n,
                )

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True}
    data_dir = '/home/robin/datasets/image_data/mnist'
    args.work_dir = '/home/robin/work_dir/llms/diffusion-toolkit/examples/work_dir/vae/'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=False,
            transform=transforms.ToTensor(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=False,
            transform=transforms.ToTensor(),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
    )
    model = VAE().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, train_loader, epoch, args)
        test(model, test_loader, epoch, args)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(args.device)
            sample = model.decode(sample).cpu()
            file_name = os.path.join(args.work_dir,
                                     'sample_' + str(epoch) + '.png')
            save_image(sample.view(64, 1, 28, 28), file_name)


if __name__ == '__main__':
    main()
