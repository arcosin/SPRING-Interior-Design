
# Imports.
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

import numpy as np
import torch

from spec import Spec
from constraint_language_v2 import *





def rnd(mi, ma):
    halfway = (ma - mi) / 2
    mu = mi + halfway
    stdev = halfway / 4#6
    s = int(round(np.random.normal(mu, stdev, size=1)[0]))
    s = max(min(s, ma), mi)
    return s



def uni(mi, ma):
    s = int(round(np.random.uniform(mi, ma, size=1)[0]))
    return s




class Synthetic_Dataset:
    '''
        Builds N new objects for training.
        Returns (location_tensors, image_tensors).
        image_tensors may be None.
    '''
    def build_object_tensor(self):
        raise NotImplementedError()
    '''
        Checks "implicit" preferences according to synthetic rules.
        Returns a boolean if all locs are correct and a list of booleans, one for each sat rule.
        Returns (None, None) if no checks are done.
    '''
    def check_preferences(self, locs):
        raise NotImplementedError()
    '''
        Convenience function to generate a default Spec that works with this dataset.
        Returns Spec class.
    '''
    @classmethod
    def default_spec(cls):
        raise NotImplementedError()






class XImplicitYExplicit(Synthetic_Dataset):
    def __init__(self, scale = (1, 1)):
        super(XImplicitYExplicit, self).__init__()
        self.cata_n = 2
        self.scale = scale

    def build_object_tensor(self, n = 1000, vecform = False):
        sx, sy = self.scale
        loc_ds = []
        for i in range(n):
            #o_1 = [rnd(0, 50 * sx), uni(40 * sy, 55 * sy), rnd(12 * sx, 25 * sx), rnd(12 * sy, 25 * sy)]
            #o_2 = [rnd(50 * sx, 100 * sx), uni(40 * sy, 55 * sy), rnd(19 * sx, 25 * sx), rnd(12 * sy, 25 * sy)]
            o_1 = [rnd(0, 500), uni(400, 550), rnd(120, 250), rnd(120, 250)]
            o_2 = [rnd(500, 1000), uni(400, 550), rnd(190, 250), rnd(120, 250)]
            locs = []
            if vecform:
                for a, b in zip(o_1, o_2):
                    locs += [a, b]
            else:
                locs += list(o_1) + list(o_2)
            loc_ds.append(torch.tensor(locs, dtype = torch.float))
        loc_ds = torch.stack(loc_ds)
        return loc_ds

    def check_preferences(self, locs):
        sx, sy = self.scale
        between = (lambda x, l, h: x >= l and x <= h)
        o_1, o_2 = [], []
        o_1 = locs[:4]
        o_2 = locs[4:8]
        prefs = [
            between(o_1[0], 0, 500),
            between(o_2[0], 500, 1000),
        ]
        return (all(prefs), prefs)

    @classmethod
    def default_spec(cls):
        #cons = [
        #    (0, "wgt", 100), (1, "wgt", 100),
        #    (0, "wlt", 256), (1, "wlt", 256),
        #    (0, "hgt", 100), (1, "hgt", 100),
        #    (0, "hlt", 256), (1, "hlt", 256),
        #    (0, 'a', 1, 300), (1, "wgt", 200), (0, "wlt", 180)
        #]
        cons = [
            con_wider_val(0, (150 / 640) * 1000, 0),
            con_wider_val(1, (150 / 640) * 1000, 0),
            con_taller_val(0, (150 / 400) * 1000, 0),
            con_taller_val(1, (150 / 400) * 1000, 0),
            con_above(0, 1, (100 / 400) * 1000),
            con_below_val(1, (50 / 400) * 1000, 0),
            con_wider_val(1, (200 / 640) * 1000, 0),
            con_narrower_val(0, (180 / 640) * 1000, 0),
            right_bound(0, 1000),
            down_bound(0, 1000),
            right_bound(1, 1000),
            down_bound(1, 1000),
        ]
        return Spec(["a white wolf", "a black wolf"], cons)







class SizeImplicitPositionExplicit(Synthetic_Dataset):
    def __init__(self, scale = (1, 1)):
        super(SizeImplicitPositionExplicit, self).__init__()
        self.cata_n = 3
        self.scale = scale

    def build_object_tensor(self, n = 1000, vecform = False):
        loc_ds = []
        sx, sy = self.scale
        for i in range(n):
            o_1 = [rnd(0, 100 * sx), rnd(30 * sy, 70 * sy), rnd(20 * sx, 25 * sx), rnd(10 * sy, 15 * sy)]   # big height, small width.
            o_2 = [rnd(0, 100 * sx), rnd(30 * sy, 70 * sy), rnd(10 * sx, 15 * sx), rnd(10 * sy, 15 * sy)]   # small height, small width.
            o_3 = [rnd(0, 100 * sx), rnd(30 * sy, 70 * sy), rnd(10 * sx, 15 * sx), rnd(20 * sy, 25 * sy)]   # small height, big width.
            locs = []
            if vecform:
                for a, b, c in zip(o_1, o_2, o_3):
                    locs += [a, b, c]
            else:
                locs += list(o_1) + list(o_2) + list(o_3)
            loc_ds.append(torch.tensor(locs, dtype = torch.float))
        loc_ds = torch.stack(loc_ds)
        return loc_ds

    def check_preferences(self, locs):
        sx, sy = self.scale
        between = (lambda x, l, h: x >= l and x <= h)
        o_1, o_2, o_3 = [], [], []
        o_1 = locs[:4]
        o_2 = locs[4:8]
        o_3 = locs[8:12]
        prefs = [
            between(o_1[2], 20 * sx, 25 * sx),
            between(o_2[2], 10 * sx, 15 * sx),
            between(o_3[2], 10 * sx, 15 * sx),
            between(o_1[3], 10 * sy, 15 * sy),
            between(o_2[3], 10 * sy, 15 * sy),
            between(o_3[3], 20 * sy, 25 * sy),
        ]
        return (all(prefs), prefs)

    @classmethod
    def default_spec(cls):
        cons = [
            (0, "wgt", 100), (1, "wgt", 100), (2, "wgt", 100),
            (0, "wlt", 256), (1, "wlt", 256), (2, "wlt", 256),
            (0, "hgt", 100), (1, "hgt", 100), (2, "hgt", 100),
            (0, "hlt", 256), (1, "hlt", 256), (2, "hlt", 256),
            (0, "wgt", 160), (0, "hgt", 160), (1, "wgt", 160),
            (1, "hgt", 160), (1, "wgt", 160), (1, "hgt", 160),
            (0, 'l', 1, 180), (1, 'l', 2, 180)
        ]
        return Spec(["a pyramid", "A cube", "A sphere"], cons)








class Compound1(Synthetic_Dataset):
    def __init__(self, scale = 1):
        super(Compound1, self).__init__()
        self.cata_n = 4
        self.scale = scale

    def build_object_tensor(self, n = 1000, vecform = False):
        loc_ds = []
        s = self.scale
        for i in range(n):
            t1 = rnd(8 * s, 12 * s)
            t2 = rnd(8 * s, 12 * s)
            t3 = rnd(8 * s, 12 * s)
            t4 = rnd(8 * s, 12 * s)
            horse_1 =  [rnd(0, 100 * s), rnd(37 * s, 56 * s), t1,  rnd(int(t1 * 1.5), 20 * s)]
            horse_2 =  [rnd(0, 100 * s), rnd(horse_1[1] - 1 * s, horse_1[1] + 1 * s), t2, rnd(int(t2 * 1.5), 20 * s)]
            tower =    [rnd(0, 90 * s),  rnd(0, 15 * s), t3, rnd(t3 * 2 - 1 * s, t3 * 2 + 1 * s)]
            farmland = [rnd(0, 100 * s), rnd(40 * s, 65 * s), t4, rnd(t4 - 1 * s, t4 + 1 * s)]
            locs = []
            if vecform:
                for a, b, c, d in zip(horse_1, horse_2, tower, farmland):
                    locs += [a, b, c, d]
            else:
                locs += list(horse_1) + list(horse_2) + list(tower) + list(farmland)
            loc_ds.append(torch.tensor(locs, dtype = torch.float))
        loc_ds = torch.stack(loc_ds)
        return loc_ds

    def check_preferences(self, locs):
        s = self.scale
        between = (lambda x, l, h: x >= l and x <= h)
        horse_1, horse_2, tower, farmland = [], [], [], []
        horse_1 = locs[:4]
        horse_2 = locs[4:8]
        tower = locs[8:12]
        farmland = locs[12:16]
        prefs = [
            between(horse_1[0], 0, 100 * s),
            between(horse_2[0], 0, 100 * s),
            between(tower[0], 0, 90 * s),
            between(farmland[0], 0, 100 * s),
            between(horse_1[1], 37 * s, 56 * s),
            between(horse_2[1], horse_1[1] - 1 * s, horse_1[1] + 1 * s),
            between(tower[1], 0, 15 * s),
            between(farmland[1], 40 * s, 65 * s),
            between(horse_1[2], 8 * s, 12 * s),
            between(horse_2[2], 8 * s, 12 * s),
            between(tower[2], 8 * s, 12 * s),
            between(farmland[2], 8 * s, 12 * s),
            between(horse_1[3], int(horse_1[2] * 1.5), 20 * s),
            between(horse_2[3], int(horse_2[2] * 1.5), 20 * s),
            between(tower[3], tower[2] * 2 - 1 * s, tower[2] * 2 + 1 * s),
            between(farmland[3], farmland[2] - 1 * s, farmland[2] + 1 * s),
        ]
        return (all(prefs), prefs)

    @classmethod
    def default_spec(cls):
        cons = [
            (0, "wgt", 100), (1, "wgt", 100), (2, "wgt", 100), (3, "wgt", 100),
            (0, "wlt", 256), (1, "wlt", 256), (2, "wlt", 256), (3, "wlt", 256),
            (0, "hgt", 100), (1, "hgt", 100), (2, "hgt", 100), (3, "hgt", 100),
            (0, "hlt", 256), (1, "hlt", 256), (2, "hlt", 256), (3, "hlt", 256),
            (0, 'l', 1, 400), (1, 'a', 3, 200 * SY), (0, "xgt", 500), (3, "wgt", 200), (3, 'r', 0, 250), (2, "ygt", 80), (2, "ylt", 150)
        ]
        return Spec(["a white horse", "a brown horse", "a fantasy tower", "A pile of colorful cubes"], cons)









class Dino(Synthetic_Dataset):
    def __init__(self, scale = 1):
        super(Dino, self).__init__()
        self.cata_n = 3

    def build_object_tensor(self, n = 1000, vecform = False):
        loc_ds = []
        for i in range(n):
            o_1 = [rnd(1, 512), rnd(100, 512), rnd(220, 256), rnd(120, 150)]   # big height, small width.
            o_2 = [rnd(1, 512), rnd(100, 512), rnd(120, 150), rnd(120, 150)]   # small height, small width.
            o_3 = [rnd(1, 512), rnd(100, 512), rnd(120, 150), rnd(220, 256)]   # small height, big width.
            locs = []
            if vecform:
                for a, b, c in zip(o_1, o_2, o_3):
                    locs += [a, b, c]
            else:
                locs += list(o_1) + list(o_2) + list(o_3)
            loc_ds.append(torch.tensor(locs, dtype = torch.float))
        loc_ds = torch.stack(loc_ds)
        return loc_ds

    def check_preferences(self, locs):
        between = (lambda x, l, h: x >= l and x <= h)
        o_1, o_2, o_3 = [], [], []
        o_1 = locs[:4]
        o_2 = locs[4:8]
        o_3 = locs[8:12]
        prefs = [
            between(o_1[0], 1, 512),
            between(o_2[0], 1, 512),
            between(o_3[0], 1, 512),
            between(o_1[1], 100, 512),
            between(o_2[1], 100, 512),
            between(o_3[1], 100, 512),
            between(o_1[2], 220, 256),
            between(o_2[2], 120, 150),
            between(o_3[2], 120, 150),
            between(o_1[3], 120, 150),
            between(o_2[3], 120, 150),
            between(o_3[3], 220, 256),
        ]
        return (all(prefs), prefs)

    @classmethod
    def default_spec(cls):
        cons = [
            (0, "wgt", 100), (1, "wgt", 100), (2, "wgt", 100),
            (0, "wlt", 256), (1, "wlt", 256), (2, "wlt", 256),
            (0, "hgt", 100), (1, "hgt", 100), (2, "hgt", 100),
            (0, "hlt", 256), (1, "hlt", 256), (2, "hlt", 256),
            (0, "l", 1, 230), (1, "a", 2, 150),
        ]
        return Spec(["a green dinosaur", "a brown dinosaur", "a blue dinosaur"], cons)






class Uniform(Synthetic_Dataset):
    def __init__(self, prompts, test_cons, bounds):
        super(Uniform, self).__init__()
        self.cata_n = len(prompts)
        self.train_constraints = []
        for i in range(len(prompts)):
            self.train_constraints += [(i, "wgt", 100), (i, "wlt", 256), (i, "hgt", 100), (i, "hlt", 256)]
        self.test_constraints = self.train_constraints + test_cons
        self.prompts = prompts
        self.bx = bounds[2]
        self.by = bounds[3]

    def build_object_tensor(self, n = 1000):
        loc_ds = []
        for i in range(n):
            locs = []
            for oi in range(len(self.prompts)):
                locs += [uni(0, self.bx), uni(0, self.by), uni(100, 256), uni(100, 256)]
            loc_ds.append(torch.tensor(locs, dtype = torch.float))
        loc_ds = torch.stack(loc_ds)
        return loc_ds

    def check_preferences(self, locs):
        return (None, None)




#===============================================================================
