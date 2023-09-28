
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as TF

from bitstring_functions import batch_number_to_bitstring, batch_bitstring_to_number, bitstring_to_number, bitstring_to_number_partial
from constraint_language_v2 import *
from reason_ledger import RLedger
from rtre_state import State
from rtre_search import Table_PCO, DFS_PCO, DFS_PCO_V2

vocab = {"left": 0, "right": 1, "end": 2, "start": 3, "pad": 4}
vocab_inverted = {v: k for k, v in vocab.items()}

vname_map = ['x', 'y', 'w', 'h']
cname_map = {"lt": "less than", "gt": "greater than", "eq": "equals"}




def wrap_sub_vec(oi, vi, omax, vmax):
    ovec = [0.0] * omax
    ovec[oi] = 1.0
    vvec = [0.0] * vmax
    vvec[vi] = 1.0
    return torch.FloatTensor(ovec + vvec).unsqueeze(0)


def ranges_intersect(a, b):
    inter = (max(a[0], b[0]), min(a[1], b[1]))
    return (inter[0] <= inter[1])





class Decoder_RNN(nn.Module):
    def __init__(self, x_size, h_size, out_size, dr = 0.2, num_rnn_layers = 3):
        super(Decoder_RNN, self).__init__()
        self.lstm = nn.GRU(x_size, h_size, num_layers = num_rnn_layers, dropout = dr)
        #self.lstm = nn.LSTM(x_size, h_size, num_layers = 3, dropout = dr)
        self.fc = torch.nn.Sequential(
            nn.Linear(h_size * 2, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
            nn.Linear(512, out_size)
        )

    def forward(self, x, h, z, force_sm = False):
        o, h = self.lstm(x.to(torch.float), h)
        o = F.leaky_relu(o)
        o = self.fc(torch.cat([o, z], dim = -1))
        if self.training and not force_sm:
            o = F.log_softmax(o.squeeze()).unsqueeze(0)
        else:
            o = F.softmax(o.squeeze()).unsqueeze(0)
        return o, h






class Sub_Encoder(nn.Module):
    def __init__(self, in_size, z_size):
        super(Sub_Encoder, self).__init__()
        self.fc = nn.Linear(in_features = in_size, out_features = z_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x




class Super_Encoder(nn.Module):
    def __init__(self, z_size, droprate = 0.2):
        super(Super_Encoder, self).__init__()
        self.fe = torch.nn.Sequential(*(list(models.resnet18(pretrained=True).children())[:-1]))
        self.append_dropout(self.fe, droprate)
        self.fc = nn.Linear(in_features = 512, out_features = z_size)

    def append_dropout(self, model, rate = 0.2):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout(module)
            if isinstance(module, nn.ReLU):
                new_model = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
                setattr(model, name, new_model)

    def forward(self, img):
        x = torch.flatten(self.fe(img), 1)
        x = F.relu(self.fc(x))
        return x







class RTRE_Reasoning_Engine(nn.Module):
    def __init__(self, cata_n, sub_in_size, z_size = 500, teach_ratio = 0.5, use_img = True, bounds = None):
        super(RTRE_Reasoning_Engine, self).__init__()
        self.cata_n = cata_n
        self.sub_size = sub_in_size
        self.z_size = z_size
        self.use_img = use_img
        self.bounds = bounds
        self.tok_size = len(vocab.keys())
        #self.system2 = Table_PCO(self, vocab, cata_n)
        self.system2 = DFS_PCO(self, vocab, cata_n)
        self.sub_encoder = Sub_Encoder(sub_in_size, z_size)
        if use_img:
            self.super_encoder = Super_Encoder(z_size, droprate = 0.2)
        else:
            self.super_encoder = None
        self.decoder = Decoder_RNN(self.tok_size, z_size, self.tok_size)
        self.teach_ratio = teach_ratio

    def obj_dict(self):
        sd = {
            "state_dict": self.state_dict(),
            "cata_n": self.cata_n,
            "sub_size": self.sub_size,
            "z_size": self.z_size,
            "use_img": self.use_img,
            "tok_size": self.tok_size,
            "teach_ratio": self.teach_ratio
        }
        return sd

    def load_obj_dict(self, obj_dict):
        self.cata_n = obj_dict["cata_n"]
        self.sub_size = obj_dict["sub_size"]
        self.z_size = obj_dict["z_size"]
        self.use_img = obj_dict["use_img"]
        self.tok_size = obj_dict["tok_size"]
        self.teach_ratio = obj_dict["teach_ratio"]
        self.load_state_dict(obj_dict["state_dict"])

    def change_bounds(self, bounds):
        self.bounds = bounds

    def encode_bg(self, img):
        if self.super_encoder is not None:
            return self.super_encoder(img)
        else:
            return None

    def trainstep(self, x, target_tokens, criterion, dev, gamma = 0.999, h_init = None, gradmask = None, debug = False):
        z = self.sub_encoder(x)
        h = torch.stack([z, z, z])
        if h_init is not None:
            h = h_init + h
        z = z.unsqueeze(0)
        if gradmask is not None:
            gm = gradmask.unsqueeze(0).unsqueeze(-1).to(dev)
        loss = 0.0
        use_teacher_forcing = True if random.random() < self.teach_ratio else False
        if use_teacher_forcing:
            t = target_tokens[0:1]
            for i in range(1, target_tokens.size(0)):
                o, h = self.decoder(t, h, z)
                targ = target_tokens[i].argmax(dim = -1)
                if debug:
                    print("      o %d:   ", o[0][0].tolist())
                    print("      t %d:   ", targ[0].tolist())
                if gradmask is not None:
                    cgm = gm.repeat(1, 1, o.size(-1))
                    o = o * cgm
                loss += criterion(o[0], targ) * (gamma ** (i - 1))
                t = target_tokens[i:i+1]
        else:
            t = target_tokens[0:1]
            for i in range(1, target_tokens.size(0)):
                o, h = self.decoder(t, h, z)
                tv, ti = o.topk(1)
                targ = target_tokens[i].argmax(dim = -1)
                if debug:
                    print("      o %d:   ", o[0][0].tolist())
                    print("      t %d:   ", targ[0].tolist())
                if gradmask is not None:
                    cgm = gm.repeat(1, 1, o.size(-1))
                    o = o * cgm
                loss += criterion(o[0], targ) * (gamma ** (i - 1))
                t = F.one_hot(ti.squeeze(-1), num_classes = len(vocab.keys())).to(dev)
        return (loss, h)

    def generate_subbitstr(self, x, dev, convert_bitstrings = True, h_init = None):
        z = self.sub_encoder(x)
        h = torch.stack([z, z, z])
        if h_init is not None:
            h = h_init + h
        z = z.unsqueeze(0)
        bitstring = []
        t = vocab["start"]
        toh = F.one_hot(torch.LongTensor([t]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
        while t != vocab["end"] and t != vocab["pad"]:
            o, h = self.decoder(toh, h, z)
            t = torch.multinomial(o[:, 0:3], 1).item()
            toh = F.one_hot(torch.LongTensor([t]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
            print("      ", o.tolist(), "   ", vocab_inverted[t])
            bitstring.append(t)
        y = bitstring
        if convert_bitstrings:
            y = bitstring_to_number(vocab, bitstring)
        return (y, h)

    def generate_raw(self, img, spec, dev):
        result = dict()
        curr_h = self.encode_bg(img)
        for oi, op in enumerate(spec.object_list):
            result[oi] = dict()
            for vi in range(4):
                vec = wrap_sub_vec(spec.object_catas[oi], vi, self.cata_n, 4).to(dev)
                gen_bitstring, curr_h = self.generate_subbitstr(vec, dev, convert_bitstrings = False, h_init = curr_h)
                val = bitstring_to_number(vocab, gen_bitstring)
                result[oi][vi] = val
        return result

    def generate_objs(self, img, bounds, spec, dev, trials_n = 1):
        ledger = RLedger()
        res, ledger = self.system2.calculate(img, bounds, spec, ledger, dev, trials_n = trials_n)
        return res

    def start_var(self, x, h, dev):
        z = self.sub_encoder(x)
        if h is None:
            h = torch.stack([z, z, z])
        else:
            h = h + torch.stack([z, z, z])
        z = z.unsqueeze(0)
        t = vocab["start"]
        toh = F.one_hot(torch.LongTensor([t]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
        init_state = State(h, z, toh, None, [])
        return init_state

    def query_bit(self, state, dev):
        h, z, tok_oh, last_out, attempted_bits = state.unpack()
        new_state = state.clone()
        if len(attempted_bits) < 1:
            o, h = self.decoder(tok_oh, h, z)
            new_state.h = h
            new_state.last_out = o
        else:
            o = last_out
            for ab in attempted_bits:
                o[0][ab] = 0.0
        if torch.all(o[:, 0:3] == 0.0):
            rem = list({vocab["left"], vocab["right"], vocab["end"]} - set(attempted_bits))
            t = rem[torch.randint(0, len(rem), (1,))]
            toh = F.one_hot(torch.LongTensor([t]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
            new_state.tok_oh = toh
            return (t, new_state)
        else:
            t = torch.multinomial(o[:, 0:3], 1).item()
            toh = F.one_hot(torch.LongTensor([t]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
            new_state.tok_oh = toh
            return (t, new_state)

    def roll_forward(self, state, bitstr, dev):
        h, z, tok_oh, last_out, _ = state.unpack()
        new_state = state.clone()
        for b in bitstr:
            o, h = self.decoder(tok_oh, h, z)
            new_state.h = h
            new_state.last_out = o
            # t = torch.multinomial(o[:, 0:3], 1).item()
            toh = F.one_hot(torch.LongTensor([b]), num_classes = len(vocab.keys())).unsqueeze(0).to(dev)
            new_state.tok_oh = toh
        return new_state




#===============================================================================
