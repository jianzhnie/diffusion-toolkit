import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import (DDPMPipeline, DDPMScheduler, UNet2DModel)
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL."""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64, filename='output.png'):
    """Given a list of PIL images, stack them together into a line for easy
    viewing."""
    output_im = Image.new('RGB', (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    output_im.save(filename)
    return output_im


def train(model, train_dataloader, optimizer, noise_scheduler, epoch, device):
    model.train()
    losses = []
    for step, batch in enumerate(train_dataloader):
        clean_images = batch['images'].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0,
                                  noise_scheduler.num_train_timesteps, (bs, ),
                                  device=clean_images.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                 timesteps)

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        print(f'Epoch:{epoch+1}, bacth: {step},  loss: {loss:.4f}')
    avg_loss = sum(losses) / len(train_dataloader)
    print(f'Epoch:{epoch+1}, train loss: {avg_loss:.4f}')

    return losses


def test(model, test_dataloader, noise_scheduler, epoch, device):
    model.eval()
    losses = []
    for step, batch in enumerate(test_dataloader):
        clean_images = batch['images'].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0,
                                  noise_scheduler.num_train_timesteps, (bs, ),
                                  device=clean_images.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                 timesteps)

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        losses.append(loss.item())
        print(f'Epoch:{epoch+1}, bacth: {step},  loss: {loss:.4f}')

    avg_loss = sum(losses) / len(test_dataloader)
    print(f'Epoch:{epoch+1}, test loss: {avg_loss:.4f}')

    return losses


def train_loop(model, train_dataloader, optimizer, noise_scheduler, epochs,
               device):
    train_losses = []
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, noise_scheduler,
                           epoch, device)
        train_losses.extend(train_loss)
    return train_losses


def main():
    # Mac users may need device = 'mps' (untested)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the butterfly pipeline
    butterfly_pipeline = DDPMPipeline.from_pretrained(
        'johnowhitaker/ddpm-butterflies-32px').to(device)

    # Create 8 images
    images = butterfly_pipeline(batch_size=8).images

    # View the result
    make_grid(images, filename="image_2.png")

    dataset = load_dataset('huggan/smithsonian_butterflies_subset',
                           split='train')
    # We'll train on 32-pixel square images, but you can try larger sizes too
    image_size = 32
    # You can lower your batch size if you're running out of GPU memory
    batch_size = 64

    # Define data augmentations
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ])

    def transform(examples):
        images = [
            preprocess(image.convert('RGB')) for image in examples['image']
        ]
        return {'images': images}

    dataset.set_transform(transform)

    # Create a dataloader from the dataset to serve up the transformed images in batches
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create a model
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 128,
                            256),  # More channels -> more parameters
        down_block_types=(
            'DownBlock2D',  # a regular ResNet downsampling block
            'DownBlock2D',
            'AttnDownBlock2D',  # a ResNet downsampling block with spatial self-attention
            'AttnDownBlock2D',
        ),
        up_block_types=(
            'AttnUpBlock2D',
            'AttnUpBlock2D',  # a ResNet upsampling block with spatial self-attention
            'UpBlock2D',
            'UpBlock2D',  # a regular ResNet upsampling block
        ),
    )
    model.to(device)

    # Set the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,
                                    beta_schedule='squaredcos_cap_v2')

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    train_losses, test_losses = train_loop(
        model,
        train_dataloader,
        optimizer,
        noise_scheduler,
        epochs=10,
        device=device,
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0].plot(train_losses)
    axs[1].plot(np.log(train_losses))
    axs[2].plot(test_losses)
    axs[3].plot(np.log(test_losses))
    plt.savefig('losses.png')


if __name__ == '__main__':
    main()
