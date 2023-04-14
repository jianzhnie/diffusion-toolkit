import open_clip
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


def guidance_generate(work_dir):

    # Prepare pretrained model
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_model.to(device)

    # Transforms to resize and augment an image + normalize to match CLIP's training data
    tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Random CROP each time
        transforms.RandomAffine(5),
        # One possible random augmentation: skews the image
        transforms.RandomHorizontalFlip(
        ),  # You can add additional augmentations if you like
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # And define a loss function that takes an image, embeds it and compares with
    # the text features of the prompt
    def clip_loss(image, text_features):
        image_features = clip_model.encode_image(tfms(image))
        # Note: applies the above transforms
        input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = F.normalize(text_features.unsqueeze(0), dim=2)
        dists = (input_normed.sub(embed_normed).norm(
            dim=2).div(2).arcsin().pow(2).mul(2))
        # Squared Great Circle Distance
        return dists.mean()

    # Prepare dataset
    prompt = "Red Rose (still life), red flower painting"  # @param
    # Explore changing this
    guidance_scale = 8  # @param
    n_cuts = 4  # @param

    image_pipe = DDPMPipeline.from_pretrained('google/ddpm-celebahq-256')
    image_pipe.to(device)
    scheduler = DDIMScheduler.from_pretrained('google/ddpm-celebahq-256')
    # More steps -> more time for the guidance to have an effect
    scheduler.set_timesteps(50)

    # We embed a prompt with CLIP as our target
    text = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(text)

    x = torch.randn(4, 3, 256, 256).to(device)
    # RAM usage is high, you may want only 1 image at a time

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        model_input = scheduler.scale_model_input(x, t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        cond_grad = 0
        for cut in range(n_cuts):
            # Set requires grad on x
            x = x.detach().requires_grad_()
            # Get the predicted x0:
            x0 = scheduler.step(noise_pred, t, x).pred_original_sample
            # Calculate loss
            loss = clip_loss(x0, text_features) * guidance_scale
            # Get gradient (scale by n_cuts since we want the average)
            cond_grad -= torch.autograd.grad(loss, x)[0] / n_cuts

        if i % 25 == 0:
            print("Step:", i, ", Guidance loss:", loss.item())

        # Modify x based on this gradient
        alpha_bar = scheduler.alphas_cumprod[i]
        x = (x.detach() + cond_grad * alpha_bar.sqrt())
        # Note the additional scaling factor here!

        # Now step with scheduler
        x = scheduler.step(noise_pred, t, x).prev_sample

    grid = torchvision.utils.make_grid(x.detach(), nrow=4)
    im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
    fig, axs = plt.subplots(1, 1, figsize=(12, 9))
    plt.imshow(im)
    fig.savefig(f'{work_dir}/clip-denoised-imgae_{i}.png')


if __name__ == '__main__':
    guidance_generate('work_dirs')