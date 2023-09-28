# Imports.
import argparse
import json
import os
from pathlib import Path
from PIL import Image

import torch
from torchvision.transforms import ToPILImage, ToTensor

from stable_diffusion_functions import build_upsampling_pipeline, upsample_image

# Constants.
NAME_STR = "upsample_images"  # Name of the module. Should use snake case (no spaces).
DESCRIP_STR = "Upsample a directory of images using a stable diffusion pipeline and save the upsampled images."  # A short description of the module and its purpose.

#---------------------------------[module code]---------------------------------


def upsample_images(input_dir, output_dir, prompt_file, sd_path, target_size, num_upsampling_steps, device):
    dev = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    upsampler = build_upsampling_pipeline(sd_path, dev)
    input_dir_path = Path(input_dir)
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    for idx, img_path in enumerate(input_dir_path.glob("*.png")):
        prompt = "a room in a house."#prompts[idx]
        print("Upscaling %s with prompt: '%s'." % (img_path, prompt))
        img = Image.open(img_path).resize((128,128))
        img_tensor = ToTensor()(img).unsqueeze(0).to(dev)
        upsampled_img_tensor = upsample_image(img_tensor, prompt, upsampler, dev, num_upsampling_steps)
        upsampled_img_pil = ToPILImage()(upsampled_img_tensor.squeeze(0).cpu())
        output_path = os.path.join(output_dir, img_path.name)
        upsampled_img_pil.save(output_path)


def main(args):
    upsample_images(args.input_dir, args.output_dir, args.prompt_file, args.sd_path, args.target_size, args.num_upsampling_steps, args.device)


#--------------------------------[module setup]---------------------------------


def config_cli_parser(parser):
    parser.add_argument("input_dir", help="Path to the directory containing the input images.")
    parser.add_argument("prompt_file", help="Path to prompt file.")
    parser.add_argument("output_dir", help="Path to the directory where the upsampled images will be saved.")
    parser.add_argument("--target_size", type=int, help="Target size of the upsampled images.", default=512)
    parser.add_argument("--num_upsampling_steps", type=int, help="Number of upsampling steps.", default=75)
    parser.add_argument("--sd_path", help="Path to stable diffusion. Only used when veg == sd.", default = "../../../Stable-Diffusion/stable-diffusion-inpainting/")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the script on (default: %(default)s).")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=NAME_STR, description=DESCRIP_STR)  # Create module's cli parser.
    parser = config_cli_parser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
