import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


def sampling_example(image_pipe, scheduler, work_dir, device):
    # The random starting point
    x = torch.randn(4, 3, 256, 256).to(
        device)  # Batch of 4, 3-channel 256 x 256 px images

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Prepare model input
        model_input = scheduler.scale_model_input(x, t)

        # Get the prediction
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)['sample']

        # Calculate what the updated sample should look like with the scheduler
        scheduler_output = scheduler.step(noise_pred, t, x)

        # Update x
        x = scheduler_output.prev_sample

        # Occasionally display both x and the predicted denoised images
        if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
            axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
            axs[0].set_title(f'Current x (step {i})')

            pred_x0 = (scheduler_output.pred_original_sample
                       )  # Not available for all schedulers
            grid = torchvision.utils.make_grid(pred_x0,
                                               nrow=4).permute(1, 2, 0)
            axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
            axs[1].set_title(f'Predicted denoised images (step {i})')
            fig.savefig(f'{work_dir}/filename_{i}.png')
    return 0


def train(image_pipe, scheduler, train_dataloader, optimizer,
          grad_accumulation_steps, epoch, device):
    losses = []
    for step, batch in tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader)):
        clean_images = batch['images'].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            image_pipe.scheduler.num_train_timesteps,
            (bs, ),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise,
                                                      timesteps)

        # Get the model prediction for the noise
        noise_pred = image_pipe.unet(noisy_images,
                                     timesteps,
                                     return_dict=False)[0]

        # Compare the prediction with the actual noise:
        loss = F.mse_loss(noise_pred, noise)
        # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)
        # Store for later plotting
        losses.append(loss.item())

        # Update the model parameters with the optimizer based on this loss
        loss.backward(loss)

        # Gradient accumulation:
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = sum(losses) / len(train_dataloader)
    print(f'Epoch {epoch} average loss: {avg_loss:.4f}')
    return losses


def train_loop(image_pipe, scheduler, train_dataloader, optimizer,
               grad_accumulation_steps, epochs, work_dirs, device):
    total_losses = []
    for epoch in range(epochs):
        losses = train(image_pipe, scheduler, train_dataloader, optimizer,
                       grad_accumulation_steps, epoch, device)
        sampling_example(image_pipe, scheduler, work_dirs, device)
        total_losses += losses
    return total_losses


def main():
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else 'cpu')
    image_pipe = DDPMPipeline.from_pretrained('google/ddpm-celebahq-256')
    image_pipe.to(device)
    # Create new scheduler and set num inference steps
    scheduler = DDIMScheduler.from_pretrained('google/ddpm-celebahq-256')
    scheduler.set_timesteps(num_inference_steps=40)

    sampling_example(
        image_pipe,
        scheduler,
        work_dir='work_dirs/fintune/',
        device=device,
    )

    # Load the dataset
    dataset_name = 'huggan/smithsonian_butterflies_subset'  # @param
    dataset = load_dataset(dataset_name, split='train')
    image_size = 256  # @param
    batch_size = 4  # @param
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform(examples):
        images = [
            preprocess(image.convert('RGB')) for image in examples['image']
        ]
        return {'images': images}

    dataset.set_transform(transform)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 2  # @param
    lr = 1e-5  # 2param
    work_dirs = 'work_dirs/'
    grad_accumulation_steps = 2  # @param
    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)
    image_pipe.scheduler = scheduler
    losses = train_loop(image_pipe=image_pipe,
                        scheduler=scheduler,
                        train_dataloader=train_dataloader,
                        optimizer=optimizer,
                        grad_accumulation_steps=grad_accumulation_steps,
                        epochs=num_epochs,
                        work_dirs=work_dirs,
                        device=device)

    # View the loss curve
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    plt.plot(losses)
    plt.ylim(0, 0.1)
    fig.savefig(f'{work_dirs}/loss.png')


if __name__ == '__main__':
    main()
