
import os
import time
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import matplotlib.pyplot as plt








class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc3(x))






class AE_Wrapper(nn.Module):
    def __init__(self, core, in_size):
        super(AE_Wrapper, self).__init__()
        self.re = core
        self.fc1 = nn.Linear(in_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.re.z_size)

    def forward(self, x, mapped = False):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        z = self.fc3(x)
        if mapped:
            xr = self.re(z, mapped = True)
            return xr
        else:
            xr = self.re(z, mapped = False)
            xr = torch.stack(list(xr)).squeeze(-1).permute(1, 0)
            return xr







def pp_to_pwh(pp):
    assert pp.size(-1) % 4 == 0
    for i in range(0, pp.size(-1), 4):
        x = pp[:, i]
        y = pp[:, i + 1]
        wx = pp[:, i + 2]
        hy = pp[:, i + 3]
        pp[:, i + 2] = wx - x
        pp[:, i + 3] = hy - y
    return pp





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






def trainstep_AE(ae, ds, opt, criterion, dev):
    real_locs = Variable(ds).to(dev)
    opt.zero_grad()
    preds = ae(real_locs)
    l = criterion(preds, real_locs)
    l.backward()
    opt.step()
    return l.item()






def trainstep_core(core, disc, opt, criterion, bs, z_size, dev):
    opt.zero_grad()
    z = torch.randn(bs, z_size).to(dev)
    valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False).to(dev)
    out_vec = core(z, mapped = False)
    loc_gen = list(out_vec)
    loc_tensor = torch.stack(loc_gen).squeeze().permute(1, 0).to(dev)
    d = disc(loc_tensor)
    l = criterion(d, valid)
    l.backward()
    opt.step()
    return l.item()






def trainstep_discriminator(core, disc, ds, opt, criterion, bs, z_size, dev):
    real_locs = Variable(ds).to(dev)
    opt.zero_grad()
    z = torch.randn(bs, z_size).to(dev)
    valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False).to(dev)
    fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False).to(dev)
    out_vec = core(z, mapped = False)
    loc_gen = list(out_vec)
    loc_tensor = torch.cat(loc_gen, dim = 1).squeeze().to(dev)
    dr = disc(real_locs)
    real_loss = criterion(dr, valid)
    dg = disc(loc_tensor.detach())
    fake_loss = criterion(dg, fake)
    l = (real_loss + fake_loss) / 2
    l.backward()
    opt.step()
    return l.item()






def checks(pref_checker, core, bs, z_size, dev):
    z = torch.randn(bs, z_size).to(dev)
    out_vec_cvx, out_vec_raw = core(z, mapped = False, use_cvx = 2)
    loc_gen_raw = list(out_vec_raw)
    loc_gen_cvx = list(out_vec_cvx)
    locs_raw = torch.stack(loc_gen_raw).squeeze().permute(1, 0).to(dev)
    locs_cvx = torch.stack(loc_gen_cvx).squeeze().permute(1, 0).to(dev)
    locs_cvx = pp_to_pwh(locs_cvx)
    ps_raw, pu_raw = 0, 0
    pref_count_raw = [0] * 16
    for i in range(locs_raw.size(0)):
        all_sat, sat_list = pref_checker(locs_raw[i])
        if all_sat:
            pref_count_raw = [xi + yi.item() for xi, yi in zip(pref_count_raw, sat_list)]
            ps_raw += 1
        else:
            pref_count_raw = [xi + yi.item() for xi, yi in zip(pref_count_raw, sat_list)]
            pu_raw += 1
    ps_cvx, pu_cvx = 0, 0
    pref_count_cvx = [0] * 16
    for i in range(locs_cvx.size(0)):
        all_sat, sat_list = pref_checker(locs_cvx[i])
        if all_sat:
            pref_count_cvx = [xi + yi.item() for xi, yi in zip(pref_count_cvx, sat_list)]
            ps_cvx += 1
        else:
            pref_count_cvx = [xi + yi.item() for xi, yi in zip(pref_count_cvx, sat_list)]
            pu_cvx += 1
    return ps_raw, pu_raw, ps_cvx, pu_cvx, pref_count_raw, pref_count_cvx



def train(ds_builder, pref_checker, core, disc, ae, z_size, dev, epochs, lr, bs, wd, test_size = 8, times = [5, 10, 15, 20, 30, 60, 90, 120, 150, 180], early_stop = False):
    criterion = nn.BCELoss()
    ae_criterion = nn.MSELoss()
    di_opt = torch.optim.Adam(disc.parameters(), lr = lr * 0.25)
    core_opt = torch.optim.Adam(core.parameters(), lr = lr)
    ae_opt = torch.optim.Adam(ae.parameters(), lr = lr)
    log_dict = {}
    losses = {"ae": [], "di": [], "ge": [], "sat_raw": [], "sat_cvx": []}
    last_age_avg = float("inf")
    age_losses = []
    timepoints = {}
    time_counter = 0
    train_start = time.time()
    test_e_labels = []
    for e in range(epochs):
        epoch_start = time.time()
        loc_ds = ds_builder(n = bs).to(dev)
        core.train()
        d_loss = trainstep_discriminator(core, disc, loc_ds, di_opt, criterion, bs, z_size, dev)
        a_loss = trainstep_AE(ae, loc_ds, ae_opt, ae_criterion, dev)
        g_loss = trainstep_core(core, disc, core_opt, criterion, bs, z_size, dev)
        losses["ae"].append(a_loss)
        losses["di"].append(d_loss)
        losses["ge"].append(g_loss)
        age_losses.append(d_loss + g_loss)
        time_so_far = time.time() - train_start
        make_trainplot(wd, losses["ae"], p = "training_plot_ae.png")
        make_trainplot(wd, losses["di"], p = "training_plot_di.png")
        make_trainplot(wd, losses["ge"], p = "training_plot_ge.png")
        print("   %d::   GL: %f   DL: %f   AL: %f   TSF: %.4f." % (e, g_loss, d_loss, a_loss, time_so_far))
        if e % 200 == 0:
            print("   Testing checkpoint.")
            test_e_labels.append(e)
            core.eval()
            with torch.no_grad():
                psr, pur, psc, puc, pcr, pcc = checks(pref_checker, core, test_size, z_size, dev)
                epoch_end = time.time()
                example_v = core.debug_vector
            losses["sat_raw"].append(sum(pcr))
            losses["sat_cvx"].append(sum(pcc))
            make_trainplot(wd, losses["sat_raw"], p = "training_plot_sat_raw.png", title = "Internal (Raw) Satisfaction", max_line = (test_size) * len(pcr), x = test_e_labels)
            make_trainplot(wd, losses["sat_cvx"], p = "training_plot_sat_cvx.png", title = "External (CVX) Satisfaction", max_line = (test_size) * len(pcc), x = test_e_labels)
            print("         p-sat-raw: %d   p-unsat-raw: %d." % (psr, pur))
            print("         p-sat-cvx: %d   p-unsat-cvx: %d." % (psc, puc))
            print("         PC-R:  %s ==> %d / %d." % (str(pcr), sum(pcr), (test_size) * len(pcr)))
            print("         PC-C:  %s ==> %d / %d." % (str(pcc), sum(pcc), (test_size) * len(pcc)))
            print("         Example v:    %s." % str(example_v[0]))
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
            if time_so_far > (60 * times[time_counter]):
                print("         Time: %d min." % times[time_counter])
                avg_sat = sum(pcr) / ((test_size) * len(pcr))
                print("         Avg sat:  %f." % avg_sat)
                timepoints[times[time_counter]] = avg_sat
                time_counter += 1
        print()
    log_dict["losses"] = losses
    log_dict["timepoints"] = timepoints
    print("Training done.")
    return log_dict

#===============================================================================
