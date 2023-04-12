import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()  # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x


def train(model, dataloader, loss_fn, optimizer, epoch, device):
    """Train the model for one epoch."""
    model.train()
    losses = []
    # Keeping a record of the losses for later viewing
    for x, y in dataloader:
        # Get some data and prepare the corrupted version
        # Data on the GPU
        x = x.to(device)
        # Pick random noise amounts
        noise_amount = torch.rand(x.shape[0]).to(device)
        # Create our noisy x
        noisy_x = corrupt(x, noise_amount)
        # Get the model prediction
        pred = model(noisy_x)
        # Calculate the loss
        loss = loss_fn(pred, x)
        # How close is the output to the true 'clean' x?
        # Backprop and update the params:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(dataloader):]) / len(dataloader)
    print(
        f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    return losses


def test(model, dataloader, epoch, work_dirs, device):
    model.eval()
    # Keeping a record of the losses for later viewing
    x, y = next(iter(dataloader))
    x = x[:8]
    # Get some data and prepare the corrupted version
    # Data on the GPU
    x = x.to(device)
    # Corrupt with a range of amounts
    amount = torch.linspace(0, 1, x.shape[0])
    # Left to right -> more corruption
    noised_x = corrupt(x, amount)
    # Get the model predictions
    with torch.no_grad():
        preds = model(noised_x.to(device)).detach().cpu()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title('Input data')
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap='Greys')
    axs[1].set_title('Corrupted data')
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1),
                  cmap='Greys')
    axs[2].set_title('Network Predictions')
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1),
                  cmap='Greys')
    fig_name = f'{work_dirs}/epoch_{epoch}.png'
    fig.savefig(fig_name)
    return 0


def generate(model, n_steps, work_dirs, epoch, device):
    x = torch.rand(8, 1, 28, 28).to(device)  # Start from random
    step_history = [x.detach().cpu()]
    pred_output_history = []

    for i in range(n_steps):
        with torch.no_grad():  # No need to track gradients during inference
            pred = model(x)  # Predict the denoised x0
        pred_output_history.append(
            pred.detach().cpu())  # Store model output for plotting
        mix_factor = 1 / (n_steps - i)
        # How much we move towards the prediction
        x = x * (1 - mix_factor) + pred * mix_factor
        # Move part of the way there
        step_history.append(x.detach().cpu())  # Store step for plotting

    fig, axs = plt.subplots(n_steps, 2, figsize=(15, 7), sharex=True)
    axs[0, 0].set_title('x (model input)')
    axs[0, 1].set_title('model prediction')
    for i in range(n_steps):
        axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(
            0, 1),
                         cmap='Greys')
        axs[i, 1].imshow(torchvision.utils.make_grid(
            pred_output_history[i])[0].clip(0, 1),
                         cmap='Greys')
    fig_name = f'{work_dirs}/denosing_{epoch}.png'
    fig.savefig(fig_name)
    return 0


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer,
               n_epochs, work_dirs, device):
    total_losses = []
    for epoch in range(n_epochs):
        losses = train(model, train_dataloader, loss_fn, optimizer, epoch,
                       device)
        test(model, test_dataloader, epoch, work_dirs, device)
        generate(model, 5, work_dirs, epoch, device)
        total_losses += losses
    return total_losses


def main(work_dirs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = torchvision.datasets.MNIST(
        root='mnist/',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(
        root='mnist/',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor())
    # Dataloader (you can mess with batch size)
    batch_size = 128
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloder = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    # How many runs through the data should we do?
    n_epochs = 5
    # Create the network
    net = BasicUNet()
    net.to(device)
    # Our loss function
    loss_fn = nn.MSELoss()
    # The optimizer
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # The training loop
    losses = train_loop(net, train_dataloader, test_dataloder, loss_fn, opt,
                        n_epochs, work_dirs, device)

    # View the loss curve
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(losses)
    plt.ylim(0, 0.1)
    fig.savefig(f'{work_dirs}/loss.png')


if __name__ == '__main__':
    main('work_dirs')
