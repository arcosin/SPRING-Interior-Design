import os
import time
import statistics

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import matplotlib.pyplot as plt

from rtre_reasoning_engine import RTRE_Reasoning_Engine, vocab
from bitstring_functions import batch_number_to_bitstring, batch_bitstring_to_number, bitstring_to_number, bitstring_to_number_partial
from bitstring_functions import bitstring_to_number




var_oh_map = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]




def reorganize_batch(x, obj_n):   # Transforms torch.Size([bs, 4 * obj_n]) to list of objects represented as list of torch.Size([bs, 1]).
    t = torch.split(x, 1, dim = -1)
    l = [list(t[n:n + 4]) for n, _ in enumerate(t) if n % 4 == 0]
    return l


def wrap_sub_vec(oi, vi, omax, vmax):
    ovec = [0.0] * omax
    ovec[oi] = 1.0
    vvec = [0.0] * vmax
    vvec[vi] = 1.0
    return torch.FloatTensor(ovec + vvec).unsqueeze(0)


def decode_rang(rang, vi):
    if isinstance(rang, tuple):
        return rang
    elif isinstance(rang, list) or isinstance(rang, dict):
        return rang[vi]
    else:
        raise ValueError("Bad input to decode_rang.")




def make_trainplot(save_dir, a_loss, p = "./training_plot.png", title = "Loss", x = None, max_line = None):
    if x is None:
        plt.plot(a_loss, color="skyblue", label="Train")
    else:
        plt.plot(x, a_loss, color="skyblue", label="Train")
    if max_line is not None:
        plt.axhline(y=max_line, color="red", linestyle='-')
    plt.legend()
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.savefig(os.path.join(save_dir, p))
    plt.clf()





def checks(pref_checker, check_list):
    ps, pu = 0, 0
    pref_count = [0] * 16
    for i in range(check_list.size(0)):
        all_sat, sat_list = pref_checker(check_list[i])
        if all_sat:
            pref_count = [xi + yi.item() for xi, yi in zip(pref_count, sat_list)]
            ps += 1
        else:
            pref_count = [xi + yi.item() for xi, yi in zip(pref_count, sat_list)]
            pu += 1
    return ps, pu, pref_count





def train_synthetic(ds_builder, pref_checker, core, cata_n, dev, epochs, lr, bs, wd, test_size = 8, times = [5, 10, 15, 20, 30, 60, 90, 120, 150, 180], print_test_objs = True, bitstr_rang = (0, 1000), early_stop = True):
    core = core.to(dev)
    opt = torch.optim.Adam(core.parameters(), lr = lr)
    criterion = nn.NLLLoss()
    train_start = time.time()
    log_dict = {}
    losses = {"loss": [], "sat_raw": []}
    last_age_avg = float("inf")
    age_losses = []
    timepoints = {}
    time_counter = 0
    test_e_labels = []
    for e in range(epochs):
        epoch_start = time.time()
        loc_ds = ds_builder(n = bs).to(dev)
        reorg_ds = reorganize_batch(loc_ds, cata_n)
        core.train()
        loss = 0.0
        curr_h = None
        for oi in range(cata_n):   #TODO: add option for permutations for object order invariance.
            for vi in range(4):
                rang = decode_rang(bitstr_rang, vi)
                vec = wrap_sub_vec(oi, vi, cata_n, 4).repeat(bs, 1).to(dev)
                targ = batch_number_to_bitstring(vocab, reorg_ds[oi][vi], rang = rang, max_cycles = 20, pad_tok = vocab["pad"])
                targ_oh = F.one_hot(targ, num_classes = len(vocab.keys())).squeeze(-2).to(dev)
                obj_loss, curr_h = core.trainstep(vec, targ_oh, criterion, dev, h_init = curr_h)
                loss += obj_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses["loss"].append(loss.item())
        age_losses.append(loss.item())
        make_trainplot(wd, losses["loss"], p = "training_plot_rnn_loss.png")
        time_so_far = time.time() - train_start
        print("   %d::   Loss: %f   TSF: %.4f." % (e, loss, time_so_far))
        if e % 200 == 0:
            print("Testing checkpoint.")
            test_e_labels.append(e)
            core.eval()
            check_lists = []
            for _ in range(test_size):
                check_list = []
                curr_h = None
                for oi in range(cata_n):
                    obj_bitstrs = []
                    obj_vals = []
                    for vi in range(4):
                        vec = wrap_sub_vec(oi, vi, cata_n, 4).to(dev)
                        gen_bitstring, curr_h = core.generate_subbitstr(vec, dev, convert_bitstrings = False, h_init = curr_h)
                        val = bitstring_to_number(vocab, gen_bitstring)
                        check_list.append(val)
                        obj_bitstrs.append(gen_bitstring)
                        obj_vals.append(val)
                    if print_test_objs:
                        print("Object %d." % oi)
                        print("  x bitstring:   %s." % str(obj_bitstrs[0]))
                        print("  y bitstring:   %s." % str(obj_bitstrs[1]))
                        print("  w bitstring:   %s." % str(obj_bitstrs[2]))
                        print("  h bitstring:   %s." % str(obj_bitstrs[3]))
                        print("  x: %d   y: %d   w: %d   h: %d." % (obj_vals[0], obj_vals[1], obj_vals[2], obj_vals[3]))
                        print()
                check_lists.append(check_list)
            check_vec = torch.tensor(check_lists).to(dev)
            ps, pu, pc = checks(pref_checker, check_vec)
            avg_sat = sum(pc) / (test_size * len(pc))
            print("     Avg sat:  %f." % avg_sat)
            losses["sat_raw"].append(sum(pc))
            make_trainplot(wd, losses["sat_raw"], p = "training_plot_rnn_sat_raw.png", title = "Internal (RNN) Satisfaction", max_line = test_size * len(pc), x = test_e_labels)
            if print_test_objs:
                print("\n\n")
            if early_stop:
                curr_age_avg = statistics.mean(age_losses)
                if curr_age_avg >= last_age_avg:
                    time_so_far = time.time() - train_start
                    print("Early stop.")
                    print("Epoch %d." % e)
                    print("Last age loss:  %f." % last_age_avg)
                    print("Curr age loss:  %f." % curr_age_avg)
                    print("Time:  %f." % time_so_far)
                    break
                else:
                    last_age_avg = curr_age_avg
            age_losses = []
        if time_counter < len(times) and time_so_far > (60 * times[time_counter]):
            print("         Time: %d min." % times[time_counter])
            avg_sat = sum(pc) / (test_size * len(pc))
            print("         Avg sat:  %f." % avg_sat)
            timepoints[times[time_counter]] = avg_sat
            time_counter += 1
    log_dict["losses"] = losses
    return log_dict






def train_coco(dl, core, cata_n, dev, epochs, lr, bs, wd, bitstr_rang = (0, 1000), early_stop = True):
    core = core.to(dev)
    opt = torch.optim.Adam(core.parameters(), lr = lr)
    criterion = nn.NLLLoss()
    train_start = time.time()
    log_dict = {}
    losses = {"loss": []}
    last_age_avg = float("inf")
    age_losses = []
    timepoints = {}
    time_counter = 0
    for e in range(epochs):
        epoch_start = time.time()
        e_flag = False
        epoch_losses = 0.0
        for img, cats, boxs in tqdm(dl):
            img = img.to(dev)
            loc_ds = boxs.to(dev)
            cats = cats.to(torch.long)
            reorg_ds = reorganize_batch(loc_ds, cats.size(-1))
            core.train()
            loss = 0.0
            curr_h = core.encode_bg(img)
            for oi in np.random.permutation(cats.size(-1)):
                cat = cats[:, oi]
                gradmask = (cat > 0).float()
                cat = F.one_hot(cat, num_classes = cata_n).squeeze(-2)
                for vi in range(4):
                    rang = decode_rang(bitstr_rang, vi)
                    ovar = torch.FloatTensor(var_oh_map[vi]).unsqueeze(0).repeat(bs, 1)
                    vec = torch.cat([cat, ovar], dim = -1).to(dev)
                    targ = batch_number_to_bitstring(vocab, reorg_ds[oi][vi], rang = rang, max_cycles = 30, pad_tok = vocab["pad"])
                    targ_oh = F.one_hot(targ, num_classes = len(vocab.keys())).squeeze(-2).to(dev)
                    if e_flag:
                        obj_loss, curr_h = core.trainstep(vec, targ_oh, criterion, dev, h_init = curr_h, gradmask = gradmask, debug = True)
                    else:
                        obj_loss, curr_h = core.trainstep(vec, targ_oh, criterion, dev, h_init = curr_h, gradmask = gradmask)
                    loss += obj_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses += loss.item()
            losses["loss"].append(loss.item())
        age_losses.append(epoch_losses)
        make_trainplot(wd, age_losses, p = "training_plot_rnn_loss.png")
        time_so_far = time.time() - train_start
        print("   %d::   Loss: %f   TSF: %.4f." % (e, epoch_losses, time_so_far))
        if e % 5 == 0:
            if early_stop:
                curr_age_avg = statistics.mean(age_losses)
                if curr_age_avg >= last_age_avg:
                    time_so_far = time.time() - train_start
                    print("Early stop.")
                    print("Epoch %d." % e)
                    print("Last age loss:  %f." % last_age_avg)
                    print("Curr age loss:  %f." % curr_age_avg)
                    print("Time:  %f." % time_so_far)
                    break
                else:
                    last_age_avg = curr_age_avg
    log_dict["losses"] = losses
    return log_dict







def test(pref_checker, core, cata_n, bs, dev, test_size = 32, print_test_objs = True):
    core = core.to(dev)
    log_dict = {}
    losses = {"loss": [], "sat_raw": []}
    core.eval()
    check_lists = []
    for _ in range(test_size):
        check_list = []
        curr_h = None
        for oi in range(cata_n):
            obj_bitstrs = []
            obj_vals = []
            for vi in range(4):
                vec = wrap_sub_vec(oi, vi, cata_n, 4).to(dev)
                gen_bitstring, curr_h = core.generate_subbitstr(vec, dev, convert_bitstrings = False, h_init = curr_h)
                val = bitstring_to_number(vocab, gen_bitstring)
                check_list.append(val)
                obj_bitstrs.append(gen_bitstring)
                obj_vals.append(val)
            if print_test_objs:
                print("Object %d." % oi)
                print("  x bitstring:   %s." % str(obj_bitstrs[0]))
                print("  y bitstring:   %s." % str(obj_bitstrs[1]))
                print("  w bitstring:   %s." % str(obj_bitstrs[2]))
                print("  h bitstring:   %s." % str(obj_bitstrs[3]))
                print("  x: %d   y: %d   w: %d   h: %d." % (obj_vals[0], obj_vals[1], obj_vals[2], obj_vals[3]))
                print()
        check_lists.append(check_list)
        check_vec = torch.tensor(check_lists).to(dev)
        ps, pu, pc = checks(pref_checker, check_vec)
        avg_sat = sum(pc) / ((bs // 4) * len(pc))
        print("     Avg sat:  %f." % avg_sat)


#===============================================================================
