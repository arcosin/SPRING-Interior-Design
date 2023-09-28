
import os, sys
import json
import argparse
import time
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, DDPMScheduler


INF_STEPS = 80


def build_stable_diffusion(sd_path, dev):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_path,
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    return pipe.to(dev)






def get_crop_box(x, y, w, h, xb, yb, n = 64):
    width = w - x                                                               # Point form to width.
    height = h - y                                                              # Point form to height.
    cx = max(0, x + (width // 2) - (n // 2))                                    # Crop x is at 0 or centered around middle of object.
    cw = cx + n                                                                 # Crop w is n past crop x.
    if cw > xb:                                                                 # Check x-axis bounds.
        cw = min(xb, w - (width // 2) + (n // 2))
        cx = cw - n
        if cx < 0:
            raise ValueError("Error: Image area is too small on x axis.")
    cy = max(0, y + (height // 2) - (n // 2))                                   # Crop y is at 0 or centered around middle of object.
    ch = cy + n                                                                 # Crop h is n past crop y.
    if ch > yb:                                                                 # Check y-axis bounds.
        ch = min(yb, h - (height // 2) + (n // 2))
        cy = ch - n
        if cy < 0:
            raise ValueError("Error: Image area is too small on y axis.")
    return (cx, cy, cw, ch)






def uncrop(outer, inner, x, y, w, h):
    outer = torch.clone(outer)
    outer[:, :, y:h, x:w] = inner
    return outer






def inpaint_odd(img, x, y, w, h, prompt, pipe, dev):
    factor = 512
    cx, cy, cw, ch = get_crop_box(x, y, w, h, img.size(3), img.size(2), n = factor)
    mask_vis = torch.ones_like(img)[:,:1]
    mask_vis = uncrop(mask_vis, 0, x, y, w, h)
    mask = torch.zeros_like(img)[:,:1]
    mask = uncrop(mask, 1, x, y, w, h)
    imgc = crop(img, cy, cx, ch - cy, cw - cx)
    maskc = crop(mask, cy, cx, ch - cy, cw - cx)
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_imgc = pipe(prompt=prompt, image=pilify(imgc[0]), mask_image=pilify(maskc[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_imgc = pipe(prompt=prompt, image=pilify(imgc[0]), mask_image=pilify(maskc[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_imgc = torchify(new_imgc).unsqueeze(0).to(dev)
    new_img = uncrop(img, new_imgc, cx, cy, cw, ch)
    applied = img * mask_vis
    obj_img = new_img[:, :, y:h, x:w]
    return (new_img, applied, mask_vis, obj_img)



def inpaint(img, x, y, w, h, prompt, pipe, dev):
    mask_vis = torch.ones_like(img)[:,:1]
    mask_vis = uncrop(mask_vis, 0, x, y, w, h)
    mask = torch.zeros_like(img)[:,:1]
    mask = uncrop(mask, 1, x, y, w, h)
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    obj_img = new_img[:, :, y:h, x:w]
    return (new_img, applied, mask_vis, obj_img)



def simple_inpaint(img, prompt, pipe, dev, center_only = True, cval = 20):
    to_size = transforms.Resize((512, 512))
    img = to_size(img)
    mask_vis = torch.ones_like(img)[:,:1]
    if center_only:
        mask_vis[:, :, cval:-cval, cval:-cval] = 0
    else:
        mask_vis[:, :, :, :] = 0
    mask = torch.zeros_like(img)[:,:1]
    if center_only:
        mask[:, :, cval:-cval, cval:-cval] = 1
    else:
        mask[:, :, :, :] = 1
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    return (new_img, applied, mask_vis)




def box_inpaint(img, prompt, pipe, dev, top_val, right_val, bottom_val, left_val):
    to_size = transforms.Resize((512, 512))
    img = to_size(img)
    mask_vis = torch.ones_like(img)[:,:1]
    mask = torch.zeros_like(img)[:,:1]
    mask[:, :, top_val:-bottom_val, left_val:-right_val] = 1
    mask_vis -= mask
    pilify = transforms.ToPILImage()
    torchify = transforms.ToTensor()
    #new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, strength=0.9, guidance_scale=15).images[0]
    new_img = pipe(prompt=prompt, image=pilify(img[0]), mask_image=pilify(mask[0]), num_inference_steps=INF_STEPS, guidance_scale=15).images[0]
    new_img = torchify(new_img).unsqueeze(0).to(dev)
    applied = img * mask_vis
    #applied[:, :, top_val:bottom_val, left_val:right_val] += new_img[:, :, top_val:bottom_val, left_val:right_val] * mask[:, :, top_val:bottom_val, left_val:right_val]
    return (new_img, applied, mask_vis)









def build_upsampling_pipeline(sd_path, dev):
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    upsampler = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        low_res_scheduler=DDPMScheduler(),
        revision="fp16",
        torch_dtype=torch.float16,
        #torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    return upsampler.to(dev)

def upsample_image(img, prompt, upsampler, dev, num_inference_steps=75):
    img_pil = to_pil_image(img[0])
    upsampled_img_pil = upsampler(
        image=img_pil,
        prompt=prompt,
        num_inference_steps=num_inference_steps
    ).images[0]
    upsampled_img_tensor = to_tensor(upsampled_img_pil).unsqueeze(0).to(dev)
    return upsampled_img_tensor


#===============================================================================
