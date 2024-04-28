import torch
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm.auto import tqdm


class ClassConditionedUnet(nn.Module):

    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning
        # information (the class embedding)
        self.model = UNet2DModel(
            sample_size=28,  # the target image resolution
            in_channels=1 + class_emb_size,
            # Additional input channels for class cond.
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                'DownBlock2D',  # a regular ResNet downsampling block
                'AttnDownBlock2D',  # a ResNet downsampling block with spatial self-attention
                'AttnDownBlock2D',
            ),
            up_block_types=(
                'AttnUpBlock2D',
                'AttnUpBlock2D',  # a ResNet upsampling block with spatial self-attention
                'UpBlock2D',  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels)  # Map to embedding dinemsion
        class_cond = class_cond.view(bs, class_cond.shape[1], 1,
                                     1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 28, 28)

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample  # (bs, 1, 28, 28)


def train(model, noise_scheduler, dataloader, loss_fn, optimizer, epoch,
          device):
    """Train the model for one epoch."""
    model.train()
    losses = []
    # Keeping a record of the losses for later viewing
    for step, (x, y) in enumerate(dataloader):
        # Get some data and prepare the corrupted version
        # Data on the GPU
        x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        # Pick random noise amounts
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0], )).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        # Get the model prediction
        pred = model(noisy_x, timesteps, y)
        # Note that we pass in the labels y
        # Calculate the loss
        loss = loss_fn(pred, x)
        # How close is the output to the true 'clean' x?
        # Backprop and update the params:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(f'Epoch {epoch}, step {step}: loss = {loss.item():05f}')
    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses) / len(dataloader)
    print(
        f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    return losses


def sampling_examples(model, noise_scheduler, epoch, work_dirs, device):
    # Prepare random x to start from, plus some desired labels y
    x = torch.randn(80, 1, 28, 28).to(device)
    y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            residual = model(x, t, y)
            # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    # Show the results
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')
    fig.savefig(f'{work_dirs}/sampling_examples_{epoch}.png')


def train_loop(model, noise_scheduler, dataloader, loss_fn, optimizer,
               n_epochs, work_dirs, device):
    """Train the model for n_epochs."""
    losses = []
    for epoch in range(n_epochs):
        losses += train(model, noise_scheduler, dataloader, loss_fn, optimizer,
                        epoch, device)
        sampling_examples(model, noise_scheduler, epoch, work_dirs, device)
    return losses


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = MNIST(root='mnist/',
                    train=True,
                    download=True,
                    transform=ToTensor())

    # Redefining the dataloader to set the batch size higher than the demo of 8
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Our network
    model = ClassConditionedUnet().to(device)
    # Create a scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule='squaredcos_cap_v2')
    # Our loss finction
    loss_fn = nn.MSELoss()
    # The optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # How many runs through the data should we do?
    n_epochs = 10
    work_dirs = 'work_dirs/condition/'
    # Train the model
    losses = train_loop(model, noise_scheduler, train_dataloader, loss_fn, opt,
                        n_epochs, work_dirs, device)

    # View the loss curve
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(losses)
    plt.ylim(0, 0.1)
    fig.savefig(f'{work_dirs}/loss.png')


if __name__ == '__main__':
    main()
