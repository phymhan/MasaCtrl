import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from masactrl.masactrl import MutualSelfAttentionControl
import fire


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


def main(
    model_path = "CompVis/stable-diffusion-v1-4",
    out_dir: str = "./workdir/masactrl_real_exp/",
    source_image_path: str = "./images/corgi.jpg",
    source_prompt = "",
    target_prompt = "a photo of a running corgi",
    scale: float = 5,
    inv_scale: float = 1,
    inv_prompt: str = 'src',
    query_intermediate: bool = False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model_path = "andite/anything-v4.0"
    # model_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)

    seed = 42
    seed_everything(seed)

    source_image = load_image(source_image_path, device)

    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}")
    os.makedirs(out_dir, exist_ok=True)

    prompts = [source_prompt, target_prompt]

    # initialize the noise map
    # invert the source image
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=inv_scale,
                                            num_inference_steps=50,
                                            return_intermediates=True)
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # results of direct synthesis
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    image_fixed = model([target_prompt],
                        latents=start_code[-1:],
                        num_inference_steps=50,
                        guidance_scale=scale)

    # inference the synthesized image with MasaCtrl
    STEP = 4
    LAYPER = 10

    # hijack the attention module
    editor = MutualSelfAttentionControl(STEP, LAYPER)
    regiter_attention_editor_diffusers(model, editor)

    if not query_intermediate:
        # inference the synthesized image
        image_masactrl = model(prompts,
                            latents=start_code,
                            guidance_scale=scale)
    else:
        # Note: querying the inversion intermediate features latents_list
        # may obtain better reconstruction and editing results
        image_masactrl = model(prompts,
                            latents=start_code,
                            guidance_scale=scale,
                            ref_intermediate_latents=latents_list)

    # save the synthesized image
    out_image = torch.cat([source_image * 0.5 + 0.5,
                        image_masactrl[0:1],
                        image_fixed,
                        image_masactrl[-1:]], dim=0)
    save_image(out_image, os.path.join(out_dir, f"all_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[0], os.path.join(out_dir, f"source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[1], os.path.join(out_dir, f"reconstructed_source_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[2], os.path.join(out_dir, f"without_step{STEP}_layer{LAYPER}.png"))
    save_image(out_image[3], os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYPER}.png"))

    print("Syntheiszed images are saved in", out_dir)


if __name__ == "__main__":
    fire.Fire(main)
