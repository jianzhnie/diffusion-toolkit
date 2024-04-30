from io import BytesIO

import requests
import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from matplotlib import pyplot as plt
from PIL import Image


# We'll use a couple of demo images later in the notebook
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')


def main() -> None:
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pipeline
    model_id = 'stabilityai/stable-diffusion-2-1-base'
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, revision='fp16', torch_dtype=torch.float16).to(device)
    pipe.enable_attention_slicing()
    # Set up a generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)

    # Run the pipeline, showing some of the available arguments
    pipe_output = pipe(
        prompt='Palette knife painting of an autumn cityscape',
        negative_prompt='Oversaturated, blurry, low quality',
        height=480,
        width=640,  # Specify the image size
        guidance_scale=8,  # How strongly to follow the prompt
        num_inference_steps=35,  # How many steps to take
        generator=generator,  # Fixed random seed
    )

    # View the resulting image:
    im = pipe_output.images[0]
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))
    axs.imshow(im)
    fig.savefig('work_dirs/stable/fig1.png')

    # Let's try a different prompt
    cfg_scales = [1.1, 8, 12]  # @param
    prompt = 'A collie with a pink hat'  # @param
    fig, axs = plt.subplots(1, len(cfg_scales), figsize=(16, 5))
    for i, ax in enumerate(axs):
        im = pipe(
            prompt,
            height=480,
            width=480,
            guidance_scale=cfg_scales[i],
            num_inference_steps=35,
            generator=generator,
        ).images[0]
        ax.imshow(im)
        ax.set_title(f'CFG Scale {cfg_scales[i]}')
    plt.savefig('work_dirs/stable/fig2.png')

    # Create some fake data (a random image, range (-1, 1))
    images = torch.rand(1, 3, 512, 512).to(device) * 2 - 1
    print('Input images shape:', images.shape)

    # Replace the scheduler
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    # Print the config
    print('Scheduler config:', pipe.scheduler)

    # Generate an image with this new scheduler
    pipe_output = pipe(
        prompt='Palette knife painting of an winter cityscape',
        height=480,
        width=480,
        generator=generator,
    )
    im = pipe_output.images[0]
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))
    axs.imshow(im)
    fig.savefig('work_dirs/stable/fig3.png')

    guidance_scale = 8  # @param
    num_inference_steps = 30  # @param
    prompt = 'Beautiful picture of a wave breaking'  # @param
    negative_prompt = 'zoomed in, blurry, oversaturated, warped'  # @param
    # Encode the prompt
    text_embeddings = pipe._encode_prompt(prompt, device, 1, True,
                                          negative_prompt)
    # Create our random starting point
    latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in enumerate(pipe.scheduler.timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(
            latent_model_input, t)

        # predict the noise residual with the unet
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text -
                                                           noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the resulting latents into an image
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())

    # View
    im = pipe.numpy_to_pil(image)[0]
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))
    axs.imshow(im)
    fig.savefig('work_dirs/stable/fig4.png')


if __name__ == '__main__':
    main()
