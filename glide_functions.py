
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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults, model_and_diffusion_defaults_upsampler

G_SCALE = 5.0


def build_glide(dev):
    has_cuda = (dev.type == 'cuda')
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(dev)
    model.load_state_dict(load_checkpoint('base-inpaint', dev))
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(dev)
    model_up.load_state_dict(load_checkpoint('upsample-inpaint', dev))
    return (model, model_up, diffusion, diffusion_up, options, options_up)



def sample_glide(prompt, model, diffusion, options, img_64, mask_64, dev):
    batch_size = 1
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, options['text_ctx'])
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])
    model_kwargs = dict(
        tokens=torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=dev),
        mask=torch.tensor([mask] * batch_size + [uncond_mask] * batch_size,dtype=torch.bool,device=dev),
        #inpaint_image=(img_64 * mask_64).repeat(full_batch_size, 1, 1, 1).to(dev),
        inpaint_image=(img_64).repeat(full_batch_size, 1, 1, 1).to(dev),
        inpaint_mask=mask_64.repeat(full_batch_size, 1, 1, 1).to(dev),
    )
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + G_SCALE * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    def denoised_fn(x_start):
        return (x_start * (1 - model_kwargs['inpaint_mask']) + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask'])
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=dev,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model.del_cache()
    return samples






def upscale_glide(prompt, samples, model_up, diffusion_up, options_up, img_256, mask_256, dev, upsample_temp = 0.997):
    batch_size = 1
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(tokens, options_up['text_ctx'])
    model_kwargs = dict(
        low_res=((samples+1)*127.5).round()/127.5 - 1,
        tokens=torch.tensor([tokens] * batch_size, device=dev),
        mask=torch.tensor([mask] * batch_size, dtype=torch.bool, device=dev),
        inpaint_image=(img_256 * mask_256).repeat(batch_size, 1, 1, 1).to(dev),
        inpaint_mask=mask_256.repeat(batch_size, 1, 1, 1).to(dev),
    )
    def denoised_fn(x_start):
        return (x_start * (1 - model_kwargs['inpaint_mask']) + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask'])
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=torch.randn(up_shape, device=dev) * upsample_temp,
        device=dev,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]
    model_up.del_cache()
    return up_samples





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


def inpaint(img, x, y, w, h, prompt, model, model_up, diffusion, diffusion_up, options, options_up, dev):
    factor = 256   #min(max(w - x, h - y) + 6, 256)
    cx, cy, cw, ch = get_crop_box(x, y, w, h, img.size(3), img.size(2), n = factor)
    mask = torch.ones_like(img)[:,:1]
    mask = uncrop(mask, 0, x, y, w, h)
    img_256 = crop(img, cy, cx, ch - cy, cw - cx)
    mask_256 = crop(mask, cy, cx, ch - cy, cw - cx)
    img_64 = F.interpolate(img_256, size=(64, 64), mode='bicubic', align_corners=False)
    mask_64 = F.interpolate(mask_256, size=(64, 64), mode="nearest")
    applied = img * mask
    sample = sample_glide(prompt, model, diffusion, options, img_64, mask_64, dev)
    upsample = upscale_glide(prompt, sample, model_up, diffusion_up, options_up, img_256, mask_256, dev)
    new_img = uncrop(img, upsample, cx, cy, cw, ch)
    return (new_img, sample, upsample, applied, mask, new_img[:, :, y:h, x:w])



def simple_inpaint(img, prompt, model, model_up, diffusion, diffusion_up, options, options_up, dev, center_only = True, cval = 20):
    img_256 = F.interpolate(img, size=(256, 256), mode='bicubic', align_corners=False)
    img_64 = F.interpolate(img_256, size=(64, 64), mode='bicubic', align_corners=False)
    mask_256 = torch.ones_like(img_256)[:,:1]
    if center_only:
        mask_256[:, :, cval:-cval, cval:-cval] = 0
    else:
        mask_256[:, :, :, :] = 0
    mask_64 = F.interpolate(mask_256, size=(64, 64), mode="nearest")
    applied = img_256 * mask_256
    sample = sample_glide(prompt, model, diffusion, options, img_64, mask_64, dev)
    upsample = upscale_glide(prompt, sample, model_up, diffusion_up, options_up, img_256, mask_256, dev)
    new_img = upsample
    return (new_img, sample, upsample, applied, mask_256)


#===============================================================================
