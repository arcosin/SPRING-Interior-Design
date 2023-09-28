
RANDOM_SEED = 13011


# Imports.
#   STL.
import os, sys
import json
import argparse
import random
import time
import statistics
from collections import deque
from PIL import Image
import warnings
warnings.simplefilter("ignore", UserWarning)
#   Non-STL.
import numpy as np
import torch
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt
#   Custom.
import synthetic_pos_datasets as syn_ds
from coco_dataset import COCO_Wrapper, collate_pad_fn, SquarePad
from coco_scanner import COCO_Scanner
from cvx_reasoning_engine import CVX_Reasoning_Engine
from cvx_training import train as train_cvx
from cvx_training import Discriminator as CVX_Disc
from cvx_training import AE_Wrapper as CVX_AE
from rtre_training import train_synthetic as train_rtre_synthetic
from rtre_training import train_coco as train_rtre_coco
from rtre_training import test as test_rtre
from glide_functions import build_glide
#build_glide = None
from glide_functions import inpaint as g_inpaint
#g_inpaint = None
from stable_diffusion_functions import build_stable_diffusion
from stable_diffusion_functions import inpaint_odd as sd_inpaint
from spec import Spec
from spec_parser import spec_parser
from bitstring_functions import batch_number_to_bitstring, batch_bitstring_to_number, bitstring_to_number, bitstring_to_number_partial
from bitstring_functions import bitstring_to_number
from rtre_reasoning_engine import RTRE_Reasoning_Engine, vocab
from constraint_language_v2 import *
from rtre_search import UnsatError


# Constants.
NAME_STR = ""
DESCRIP_STR = ""

DEF_BS = 8
DEF_LR = 0.0001
DEF_Z = 500
DEF_EPOCHS = 1000
TEACH_RATIO = 0.75


ds_names = ["syn_simple", "syn_tight", "syn_complex_1", "syn_dino", "coco"]


def build_ds(args):
    name = args.ds
    if name == "syn_simple":
        ds = syn_ds.XImplicitYExplicit(scale = (1.0, 1.0))
    elif name == "syn_tight":
        ds = syn_ds.SizeImplicitPositionExplicit(scale = (1.0, 1.0))
    elif name == "syn_complex_1":
        ds = syn_ds.Compound1(scale = 1.0)
    elif name == "syn_dino":
        ds = syn_ds.Dino()
    elif name == "coco":
        if args.re != "rtre":
            raise NotImplementedError("COCO is not implemented for this reasoning engine.")
        #ds = COCO_Wrapper.from_args_animals(args.coco_img_dir, args.coco_json, args.training_bg_dir)
        ds = COCO_Wrapper.from_args_interior(args.coco_img_dir, args.coco_json, args.training_bg_dir)
    return ds

def default_cons_sd(objs, bx = 1000, by = 1000):
    cons = []
    for i in objs:
        cons.append(con_wider_val(i, (250 / bx) * 1000, 0))
        cons.append(con_narrower_val(i, (512 / bx) * 1000, 0))
        cons.append(con_taller_val(i, (250 / by) * 1000, 0))
        cons.append(con_shorter_val(i, (512 / by) * 1000, 0))
        cons.append(right_bound(i, bx))
        cons.append(down_bound(i, by))
    return cons

def default_cons_glide(objs, bx = 1000, by = 1000):
    cons = []
    for i in objs:
        cons.append(con_wider_val(i, (128 / bx) * 1000, 0))
        cons.append(con_narrower_val(i, (256 / bx) * 1000, 0))
        cons.append(con_taller_val(i, (128 / by) * 1000, 0))
        cons.append(con_shorter_val(i, (256 / by) * 1000, 0))
        cons.append(right_bound(i, bx))
        cons.append(down_bound(i, by))
    return cons

def build_programmed_spec(name, defcon_builder = default_cons_sd):
    if name == "syn_simple":
        spec = syn_ds.XImplicitYExplicit.default_spec()
    elif name == "syn_tight":
        spec = syn_ds.SizeImplicitPositionExplicit.default_spec()
    elif name == "syn_complex_1":
        spec = syn_ds.Compound1.default_spec()
    elif name == "coco":
        # NOTE: Replace this with desired testing spec.
        cons = defcon_builder([0,1])
        cons += [
                con_below_val(0, 200, 0), con_below_val(1, 200, 0),
                con_aboveabove(0, 1, 0), con_taller(1,0,0),
                ConstraintOR([con_rightright(0, 1, 0), con_leftleft(0, 1, 0)])
               ]
        catas = [12, 13]
        prompts = ["A black microwave", "A nice looking cooking oven"]
        spec = Spec(prompts, cons, catas)
    else:
        raise NotImplementedError("No default spec for this dataset.")
    return spec


def build_programmed_spec_scan(name, defcon_builder, scan_data):
    i = 0
    object_list = []
    painteds = []
    object_catas = []
    paint_flags = []
    cons = []
    for sco in scan_data:
        print("Found scanned object %d, category %d." % (i, sco["cata"]))
        print((sco["box"][0], sco["box"][1], sco["box"][2], sco["box"][3]))
        object_list.append("?")
        object_catas.append(sco["cata"])
        paint_flags.append(False)
        cons += set_loc(i, sco["box"][0], sco["box"][1], sco["box"][2], sco["box"][3])
        i += 1
        if len(object_list) >= 3:
            break
    # NOTE: Replace this with desired testing spec.
    cons += defcon_builder([i, i+1])
    cons += [
        ConstraintOR([
            ConstraintAND(cons_atop(i, 1, 200, 180) + cons_atop(i+1, 0, 200, 180)),
            ConstraintAND(cons_atop(i, 2, 200, 180) + cons_atop(i+1, 1, 200, 180)),
        ]),
        con_leftleft(i+1, i,0),
        con_narrower(i, 0, 0),
        con_taller_val(i, 300),
        con_shorter(i+1, i, 75),
        con_narrower(i+1, 2, 0),
    ]
    object_catas += [3, 3]
    object_list += ["A big red flower in a container", "A green leafy fern in a container"]
    paint_flags+= [True, True, True]
    return Spec(object_list, cons, object_catas=object_catas, paint_flags=paint_flags)

def build_scanner(args):
    name = args.ds
    if name == "coco":
        return COCO_Scanner(sensitivity = args.scan_sensitivity, categories = [62,63,64,65,66,67,68,69,70,71,72,78,79,80,81,82,83])
    else:
        raise NotImplementedError("No scanner for this dataset.")

def pil_to_torch(pil_img, neg_mode = False):
    img = np.array(pil_img)
    if neg_mode:
        img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 127.5 - 1     # (-1, 1).
    else:
        img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0         # (0, 1).
    return img_tensor

def read_image(p, size = None):
    pil_img = Image.open(p).convert('RGB')
    if size is not None:
        if isinstance(size, float):
            pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
        else:   # Assumed tuple.
            pil_img = pil_img.resize(size, resample=Image.BICUBIC)
    return pil_img

def rel_to_pix(loc_map, image_dimms):
    for oi in loc_map.keys():
        for vi in loc_map[oi].keys():
            if vi % 2 == 0:   # X or W.
                loc_map[oi][vi] = int(round((loc_map[oi][vi] / 1000) * image_dimms[0]))
            else:   # Y or H.
                loc_map[oi][vi] = int(round((loc_map[oi][vi] / 1000) * image_dimms[1]))
    return loc_map

def pix_to_rel(loc_map, image_dimms):
    for oi in loc_map.keys():
        for vi in loc_map[oi].keys():
            if vi % 2 == 0:   # X or W.
                loc_map[oi][vi] = int(round((loc_map[oi][vi] / image_dimms[0]) * 1000))
            else:   # Y or H.
                loc_map[oi][vi] = int(round((loc_map[oi][vi] / image_dimms[1]) * 1000))
    return loc_map

def pix_to_rel_list(loc_list, image_dimms):
    for vi in range(len(loc_list)):
        if vi % 2 == 0:   # X or W.
            loc_list[vi] = int(round((loc_list[vi] / image_dimms[0]) * 1000))
        else:   # Y or H.
            loc_list[vi] = int(round((loc_list[vi] / image_dimms[1]) * 1000))
    return loc_list

def print_train_setup(args):
    print("   Z_size:  %d." % args.zs)
    print("   Batch size:  %d." % args.bs)
    print("   Learning rate:  %f." % args.lr)
    print("   Epochs:  %f." % args.epochs)
    print("   Dataset:  %s." % args.ds)

def read_bg(p, new_size = None):
    bg_img = read_image(p, size=new_size)
    bounds = [0, 0, bg_img.size[0], bg_img.size[1]]
    print("   Bounds vector of the image:  %s." % str(bounds))
    return (bg_img, bounds)

def load_core(args, dev):
    if args.re == "npre":
        obj_dict = torch.load(os.path.join(args.wd, "core.pt"))
        core = CVX_Reasoning_Engine(obj_dict["z_size"], obj_dict["num_objs"])
        core.load_obj_dict(obj_dict)
    elif args.re == "rtre":
        obj_dict = torch.load(os.path.join(args.wd, "core.pt"))
        core = RTRE_Reasoning_Engine(obj_dict["cata_n"], obj_dict["sub_size"], z_size = obj_dict["z_size"], teach_ratio = obj_dict["teach_ratio"], use_img = obj_dict["use_img"])
        core.load_obj_dict(obj_dict)
    return core.to(dev)

def save_core(args, core, dev):
    core.to("cpu")
    torch.save(core.obj_dict(), os.path.join(args.wd, "core.pt"))
    core.to(dev)



def train(args, dev):
    ds = build_ds(args)
    cata_n = ds.cata_n
    z_size = args.zs
    print_train_setup(args)
    bounds = [0, 0, args.train_xb, args.train_yb]
    if args.re == "npre":
        bounds = torch.FloatTensor(bounds).to(dev)
        core = CVX_Reasoning_Engine(z_size, cata_n, constraint_list = [], x_size = 0, bounds = bounds).to(dev)
        disc = CVX_Disc(cata_n * 4).to(dev)
        ae = CVX_AE(core, cata_n * 4).to(dev)
        train_start = time.time()
        log_dict = train_cvx(ds.build_object_tensor,
                             ds.check_preferences,
                             core, disc, ae,
                             z_size,
                             dev,
                             args.epochs,
                             args.lr,
                             args.bs,
                             args.wd)
        train_t = time.time() - train_start
    elif args.re == "rtre":
        if args.ds == "coco":
            print("Dataset size: %d." % len(ds))
            core = RTRE_Reasoning_Engine(cata_n, 4 + cata_n, z_size = z_size, teach_ratio = TEACH_RATIO, use_img = True)
            train_start = time.time()
            dl = DataLoader(ds, collate_fn=collate_pad_fn, batch_size=args.bs, drop_last=True, shuffle=True)
            log_dict = train_rtre_coco(dl,
                                       core,
                                       cata_n,
                                       dev,
                                       args.epochs,
                                       args.lr,
                                       args.bs,
                                       args.wd)
            train_t = time.time() - train_start
        else:
            core = RTRE_Reasoning_Engine(cata_n, 4 + cata_n, z_size = z_size, teach_ratio = TEACH_RATIO, use_img = False)
            train_start = time.time()
            log_dict = train_rtre_synthetic(ds.build_object_tensor,
                                            ds.check_preferences,
                                            core,
                                            cata_n,
                                            dev,
                                            args.epochs,
                                            args.lr,
                                            args.bs,
                                            args.wd,
                                            print_test_objs = True)
            train_t = time.time() - train_start
    print("   Training complete.\n   Training time:  %s." % str(train_t))
    return core




def test(args, core, dev):
    ds = build_ds(args)
    cata_n = ds.cata_n
    bounds = [0, 0, args.train_xb, args.train_yb]
    if args.re == "npre":
        raise NotImplementedError()
    elif args.ds == "coco":
        raise NotImplementedError("Explicit preference checking is not available for this dataset.")
    elif args.re == "rtre":
        test_start = time.time()
        test_rtre(ds.check_preferences,
                  core,
                  cata_n,
                  args.bs,
                  dev,
                  print_test_objs = True)
        test_t = time.time() - test_start
    print("   Training complete.\n   Training time:  %s." % str(test_t))
    return core




def generate_scene_cvx(args, core, glide_model_tuple, bg_img, bounds, spec, dev, postproc = None):
    bounds = torch.FloatTensor(bounds).to(dev)
    core.change_constraints(constraint_list = spec.constraint_list, bounds = bounds)
    prompts = spec.object_list
    model, model_up, diffusion, diffusion_up, options, options_up = glide_model_tuple
    for e in range(args.examples):
        print("   Running placer, example %d." % e)
        img = pil_to_torch(bg_img)
        z = torch.randn(1, core.z_size).to(dev)
        place_list = core(z, use_cvx = True)
        place_list = [p.long() for p in place_list]
        place_map = []
        masked = img.clone()
        for i in range(0, len(place_list), 4):
            place_map.append(place_list[i: i + 4])
        for o in range(len(prompts)):
            x, y, w, h = place_map[o]
            prompt = prompts[o]
            print("   Placing '%s' at coords (x, y, w, h):   (%d, %d, %d, %d)." % (prompt, x, y, w, h))
            img, img_64, img_256, applied, m = inpaint(img, x, y, w, h, prompt, model, model_up, diffusion, diffusion_up, options, options_up, dev)
            masked = masked * m
            print("   Object placed.")
            if args.show_steps:
                save_image(applied, os.path.join(args.wd, "example_%d_mask_%d.png" % (e, o)), normalize=True)
                save_image(img, os.path.join(args.wd, "example_%d_img_%d.png" % (e, o)), normalize=True)
                save_image(img_64, os.path.join(args.wd, "example_%d_img_64_%d.png" % (e, o)), normalize=True)
                save_image(img_256, os.path.join(args.wd, "example_%d_img_256_%d.png" % (e, o)), normalize=True)
        if postproc is not None:
            img = postproc(img)
        save_image(img, os.path.join(args.wd, "example_%d_final.png" % e), normalize=True)
        save_image(masked, os.path.join(args.wd, "example_%d_masks.png" % e), normalize=True)




def generate_scene_rtre(args, core, veg, bg_img, bounds, spec, dev, fn_post = "", save_singles = True, postproc = None, postproc_singles = None, redo_unsat = True):
    core.eval()
    img_t = T.Compose([SquarePad(), T.Resize((128, 128)), T.ToTensor()])
    bounds_tensor = torch.FloatTensor(bounds).to(dev)
    prompts = spec.object_list
    paint_flags = spec.paint_flags
    for e in range(args.examples):
        start_time = time.time()
        print("   Running placer, example %d." % e)
        img_in = img_t(bg_img).unsqueeze(0).to(dev)
        img = pil_to_torch(bg_img, (args.veg == "glide")).to(dev)
        print("before"); print(img.min()); print(img.max())
        if redo_unsat:
            redo = True
            while redo:
                try:
                    place_map = core.generate_objs(img_in, ((0, 1000), (0, 1000), (0, 1000), (0, 1000)), spec, dev, trials_n = 1)
                    redo = False
                except UnsatError:
                    pass
        else:
            place_map = core.generate_objs(img_in, ((0, 1000), (0, 1000), (0, 1000), (0, 1000)), spec, dev, trials_n = 1)
        end_time = time.time()
        print("placement map (rel):")
        print(place_map)
        place_map = rel_to_pix(place_map, (bounds[2], bounds[3]))
        print("placement map (pix):")
        print(place_map)
        for o in range(len(prompts)):
            place_map[o][2] = place_map[o][2] + place_map[o][0]
            place_map[o][3] = place_map[o][3] + place_map[o][1]
        print("placement map (point-point):")
        print(place_map)
        print("Time taken: %f." % (end_time - start_time))
        masked = img.clone()
        if save_singles:
            if not os.path.isdir(os.path.join(args.wd, "obj")):
                os.mkdir(os.path.join(args.wd, "obj"))
        for o in range(len(prompts)):
            if paint_flags[o]:
                x = place_map[o][0]
                y = place_map[o][1]
                w = place_map[o][2]
                h = place_map[o][3]
                prompt = prompts[o]
                print("   Placing '%s' at coords (x, y, w, h):   (%d, %d, %d, %d)." % (prompt, x, y, w, h))
                if args.veg == "glide":
                    model, model_up, diffusion, diffusion_up, options, options_up = veg
                    img, img_64, img_256, applied, m, o_img = g_inpaint(img, x, y, w, h, prompt, model, model_up, diffusion, diffusion_up, options, options_up, dev)
                else:
                    img, applied, m, o_img = sd_inpaint(img, x, y, w, h, prompt, veg, dev)
                masked = masked * m
                print("   Object placed.")
                if save_singles:
                    if postproc_singles is not None:
                        save_image(postproc_singles(o_img), os.path.join(args.wd, "obj", "example_%d_obj_%d%s.png" % (e, o, fn_post)), normalize=True)
                    else:
                        save_image(o_img, os.path.join(args.wd, "obj", "example_%d_obj_%d%s.png" % (e, o, fn_post)), normalize=True)
                if args.show_steps:
                    save_image(applied, os.path.join(args.wd, "example_%d_mask_%d.png" % (e, o)), normalize=True)
                    save_image(img, os.path.join(args.wd, "example_%d_img_%d.png" % (e, o)), normalize=True)
            else:
                x = place_map[o][0]
                y = place_map[o][1]
                w = place_map[o][2]
                h = place_map[o][3]
                print("   Acknowledging object at coords (x, y, w, h):   (%d, %d, %d, %d)." % (x, y, w, h))
        if not os.path.isdir(os.path.join(args.wd, "gen")):
            os.mkdir(os.path.join(args.wd, "gen"))
        if not os.path.isdir(os.path.join(args.wd, "masks")):
            os.mkdir(os.path.join(args.wd, "masks"))
        if postproc is not None:
            save_image(postproc(img), os.path.join(args.wd, "gen", "example_%d_final%s.png" % (e, fn_post)), normalize=True)
        else:
            save_image(img, os.path.join(args.wd, "gen", "example_%d_final%s.png" % (e, fn_post)), normalize=True)
        print("Saved in %s." % os.path.join(args.wd, "gen", "example_%d_final%s.png" % (e, fn_post)))
        save_image(masked, os.path.join(args.wd, "masks", "example_%d_masks%s.png" % (e, fn_post)), normalize=True)




def main(args):
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:%d" % args.gpu_id)
    print("Device:  %s." % dev)
    if args.mode == "train":
        print("Building and training core model.")
        core = train(args, dev)
        print("Saving core model.")
        save_core(args, core, dev)
    elif args.mode == "eval":
        print("Loading core model.")
        core = load_core(args, dev)
        test(args, core, dev)
    elif args.mode in ["demo", "demo_parse", "demo_parse_scan", "demo_scan"]:
        print("Loading core model.")
        core = load_core(args, dev)
        print("Building VEG model.")
        if args.veg == "glide":
            veg = build_glide(dev)   # (model, model_up, diffusion, diffusion_up, options, options_up)
            defcon_builder = default_cons_glide
        else:
            veg = build_stable_diffusion(args.sd_path, dev)
            defcon_builder = default_cons_sd
        print("Reading background image.")
        if args.resize:
            bg_img, bounds = read_bg(args.bg_img, (args.resize_x, args.resize_y))
        else:
            bg_img, bounds = read_bg(args.bg_img)
        if args.mode in ["demo_parse_scan", "demo_scan"]:
            print("Scanning background image.")
            scanner = build_scanner(args)
            scan_data = scanner(bg_img)
            for sco in scan_data:
                print("Scanned box (pix):  %s." % str(sco["box"]))
                sco["box"] = pix_to_rel_list(sco["box"], (bounds[2], bounds[3]))
                print("Scanned box (rel):  %s." % str(sco["box"]))
        else:
            scan_data = []
        if args.mode == "demo":
            print("Using default dataset spec.")
            spec = build_programmed_spec(args.ds, defcon_builder)
        elif args.mode == "demo_scan":
            print("Using default dataset spec.")
            spec = build_programmed_spec_scan(args.ds, defcon_builder, scan_data)
        else:
            print("Starting spec interface.")
            cata_dict = build_ds(args).label_dict()   # NOTE: Fix this later.
            spec = spec_parser(cata_dict,
                               scan_data = scan_data,
                               defcon_builder = defcon_builder,
                               read_file = args.spec_infile,
                               defcon_bounds = (bounds[2], bounds[3]))
        print("Generating %d examples." % args.examples)
        if args.re == "npre":
            generate_scene_cvx(args, core, veg, bg_img, bounds, spec, dev)
        else:
            generate_scene_rtre(args, core, veg, bg_img, bounds, spec, dev)
    print("Done.")





#--------------------------------[module setup]---------------------------------

def config_cli_parser(parser):
    parser.add_argument("--mode", help="Runmode of the application.", choices = ["train", "eval", "demo", "demo_scan", "demo_parse", "demo_parse_scan"], default = "demo")
    parser.add_argument("--re", help="Reasoning engine to use.", choices = ["rtre", "npre"], default = "rtre")
    parser.add_argument("--cpu", help="Specify whether the CPU-only should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--gpu_id", help="Device ID for GPU to use if running on CUDA.", type=int, default=0)
    parser.add_argument("--wd", help="Working directory for saving all logs, output data, and models.", default = "./logs/")
    parser.add_argument("--veg", help="Which visual element generator to use.", choices = ["glide", "sd"], default = "sd")
    # Generation.
    parser.add_argument("--bg_img", help="Background image to use in generation.", default = "./bgs/bg1.png")
    parser.add_argument("--examples", help="Examples to show in demo mode or at the end of train mode.", type = int, default = 1)
    parser.add_argument("--show_steps", help="Specify whether to show each scene generation step.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--coco_img_dir", help="COCO dataset image directory. Only used if coco is the selected dataset.", default = "../../../Shared_Datasets/coco/train/train2017/")
    parser.add_argument("--coco_json", help="COCO dataset json annotation file. Only used if coco is the selected dataset.", default = "../../../Shared_Datasets/coco/annot/instances_train2017.json")
    parser.add_argument("--sd_path", help="Path to stable diffusion. Only used when veg == sd.", default = "../../../Stable-Diffusion/stable-diffusion-inpainting/")
    parser.add_argument("--spec_infile", help="File read like standard input for spec. Optional.", type = str, default = None)
    parser.add_argument("--resize", help="Whether the background should be resized.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--resize_x", help="Resize pixels for x dimmension.", type = int, default = 1000)
    parser.add_argument("--resize_y", help="Resize pixels for y dimmension.", type = int, default = 1000)
    # Train only.
    parser.add_argument("--ds", help="Dataset to train on.", choices = ds_names, default = ds_names[-1])
    parser.add_argument("--epochs", help="Epochs to train for.", type = int, default = DEF_EPOCHS)
    parser.add_argument("--zs", help="Size of the Z (noise) dimmension.", type = int, default = DEF_Z)
    parser.add_argument("--lr", help="Learning rate.", type = float, default = DEF_LR)
    parser.add_argument("--bs", help="Batch size during training.", type = int, default = DEF_BS)
    parser.add_argument("--training_bg_dir", help="COCO backgrounds dir for training.", default = None)
    parser.add_argument("--train_xb", help="The bound used during training for x axis values.", type = int, default = 1000)
    parser.add_argument("--train_yb", help="The bound used during training for y axis values.", type = int, default = 1000)
    # Scan.
    parser.add_argument("--scan_sensitivity", help="Sensitivity of the scanner.", type = float, default = 0.70)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = NAME_STR, description = DESCRIP_STR)   # Create module's cli parser.
    parser = config_cli_parser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
