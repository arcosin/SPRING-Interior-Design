
import os, sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import PIL
from PIL import Image

'''
    COCO categories:
        1 -- human
        18 -- dog
        19 -- horse
        20 -- sheep
        21 -- cow
        22 -- elephant
        23 -- bear
        24 -- zebra
        25 -- giraffe
'''


'''
    This function is for replacing a method in a library object.
    This practice is big bad, but speeds this up allot.
'''
def replacment_func(self, id):
    return None


# 62,63,64,65,66,67,68,69,70,71,72,78,79,80,81,82,83
cocoid_to_name = {
62: "chair",
63: "couch",
64: "potted plant",
65: "bed",
66: "mirror",
67: "dining table",
68: "window",
69: "desk",
70: "toilet",
71: "door",
72: "tv",
78: "microwave",
79: "oven",
80: "toaster",
81: "sink",
82: "refrigerator",
83: "blender"
}



def scale(v, mx, my, sx = 256, sy = 256):
     v[0] = int(round(v[0] / mx * sx))
     v[1] = int(round(v[1] / my * sy))
     v[2] = int(round(v[2] / mx * sx))
     v[3] = int(round(v[3] / my * sy))
     return v


class SquarePad:
    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            w, h = image.size
        else:
            w = image.size(-2)
            h = image.size(-1)
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, 'constant')




class COCO_Wrapper(Dataset):
    def __init__(self, coco, bg_dir = None, categories = [18, 19, 20, 21, 22, 23, 25], cata_labels = ["dog", "horse", "sheep", "cow", "elephant", "bear", "giraffe"]):
        self.categories = [None] + categories
        self.category_labels = ["<pad>"] + cata_labels
        self.cata_n = len(self.categories)
        self.max_dims = (640, 640)
        inds_to_keep = []
        # NOTE: big bad but dont touch.
        orig_method = coco._load_image
        coco._load_image = types.MethodType(replacment_func, coco)
        for i, (img, targ) in enumerate(coco):
            if len(targ) > 5:
                continue
            for t in targ:
                if t["iscrowd"] == 0 and t["category_id"] in categories:
                    inds_to_keep.append(i)
                    break
        coco._load_image = orig_method
        # NOTE: end of big bad.
        self.coco = Subset(coco, inds_to_keep)
        self.bg_dir = bg_dir
        self.img_t = transforms.Compose([SquarePad(), transforms.Resize((128, 128)), transforms.ToTensor()])

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, targ = self.coco.__getitem__(idx)
        cats = []
        boxs = []
        for t in targ:
            if t["iscrowd"] == 0 and t["category_id"] in self.categories:
                cats.append(self.categories.index(t["category_id"]))
                iw, ih = img.size
                max_wh = np.max([iw, ih])
                hp = int((((max_wh - iw) / max_wh) * 1000) / 2)
                vp = int((((max_wh - ih) / max_wh) * 1000) / 2)
                b = list(t["bbox"])
                b = scale(b, iw, ih, 1000 - 2*hp, 1000 - 2*vp)
                b = [b[0] + hp, b[1] + vp, b[2], b[3]]
                boxs.append(b)
        if self.bg_dir is not None:
            img_id = str(t["image_id"]).zfill(12)
            bg_path = os.path.join(self.bg_dir, "%s.jpg" % img_id)
            img = Image.open(bg_path).convert("RGB")
        return (self.img_t(img), cats, boxs)

    def id_to_coco_id(self, id):
        return self.categories[id]

    def label_dict(self):
        d = {c: l for c, l in zip(range(len(self.categories)), self.category_labels)}
        del d[0]
        return d

    @classmethod
    def from_args(cls, coco_dir, coco_json, bg_dir = None):
        ds = dset.CocoDetection(root = coco_dir, annFile = coco_json)
        return cls(ds, bg_dir = bg_dir)

    @classmethod
    def from_args_animals(cls, coco_dir, coco_json, bg_dir = None):
        categories = [18, 19, 20, 21, 22, 23, 25]
        cata_labels = ["dog", "horse", "sheep", "cow", "elephant", "bear", "giraffe"]
        ds = dset.CocoDetection(root = coco_dir, annFile = coco_json)
        return cls(ds, bg_dir = bg_dir, categories=categories, cata_labels=cata_labels)

    @classmethod
    def from_args_interior(cls, coco_dir, coco_json, bg_dir = None):
        categories = [62,63,64,65,66,67,68,69,70,71,72,78,79,80,81,82,83]
        cata_labels = [ "chair",
                        "couch",
                        "potted plant",
                        "bed",
                        "mirror",
                        "dining table",
                        "window",
                        "desk",
                        "toilet",
                        "door",
                        "tv",
                        "microwave",
                        "oven",
                        "toaster",
                        "sink",
                        "refrigerator",
                        "blender"]
        #categories = [62]
        #cata_labels = ["chair"]
        ds = dset.CocoDetection(root = coco_dir, annFile = coco_json)
        return cls(ds, bg_dir = bg_dir, categories=categories, cata_labels=cata_labels)




def collate_pad_fn(data):
    max_len = 0
    imgs, catss, boxss = [], [], []
    for single in data:
        max_len = max(max_len, len(single[1]))
    for i in range(len(data)):
        img, cats, boxs = data[i]
        while len(cats) < max_len:
            cats.append(0.0)
            boxs.append([0.0] * 4)
        boxs = [x for li in boxs for x in li]
        boxs = torch.tensor(boxs)
        cats = torch.tensor(cats)
        catss.append(cats)
        boxss.append(boxs)
        imgs.append(img)
    return (torch.stack(imgs), torch.stack(catss), torch.stack(boxss))

'''
img_dir = "./Shared_Datasets/coco/train/train2017/"
annot_file = "./Shared_Datasets/coco/annot/instances_train2017.json"
ds = dset.CocoDetection(root = img_dir, annFile = annot_file)
dsw = COCO_Wrapper(ds)
dl = DataLoader(dsw, collate_fn=collate_pad_fn, batch_size=5)
next(iter(dl))[1]
'''
#===============================================================================
