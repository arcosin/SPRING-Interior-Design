import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.transforms.functional as TF

from bitstring_functions import *
from constraint_language_v2 import *
from possibility_table import *
from spec import Spec
from rtre_state import State



def wrap_sub_vec(oi, vi, omax, vmax):
    ovec = [0.0] * omax
    ovec[oi] = 1.0
    vvec = [0.0] * vmax
    vvec[vi] = 1.0
    return torch.FloatTensor(ovec + vvec).unsqueeze(0)


'''
    Abstract class for RTRE positional constraint optimizers.
'''
class RTRE_PCO(object):
    def calculate(self, img, bounds, spec, dev):
        raise NotImplementedError()



class UnsatError(Exception):
    pass





class Table_PCO(RTRE_PCO):
    def __init__(self, rtre, vocab, cata_n):
        super(Table_PCO, self).__init__()
        self.vocab = vocab
        self.rtre = rtre
        self.cata_n = cata_n

    def _handle_deterministic(self, curr_bitstr, val, state, dev):
        final_bitstr = number_to_bitstring(self.vocab, val)
        so_far = len(curr_bitstr)
        if so_far >= len(final_bitstr):
            return state
        else:
            assert curr_bitstr == final_bitstr[:so_far]
            return self.rtre.roll_forward(state, final_bitstr[so_far:], dev)

    def calculate(self, img, bounds, spec, ledger, dev, trials_n = 1):
        possible = poss_table(spec, bounds)
        possible, viol_con = update_poss_table(possible, spec, ledger)  # Update for deterministic constraints / bounds.
        curr_h = self.rtre.encode_bg(img)                                            # Init hidden vector with encoded bg image.
        #for oi, oc in enumerate(spec.scan_catas):                               # Iterate through objects.
        #    for vi in range(4):                                                 # Iterate through variables [x,y,w,h].
        #        x = wrap_sub_vec(spec.object_catas[oi], vi, self.cata_n, 4).to(dev)   # Wrap (oi, vi) vector.
        #        state = self.rtre.start_var(x, curr_h, dev)                     # Init var gen session.
        #        state = self._handle_deterministic([], self.scan_values[oi][vi], state, dev)
        #        curr_h = state.h
        for oi, op in enumerate(spec.object_list):                              # Iterate through objects.
            for vi in range(4):                                                 # Iterate through variables [x,y,w,h].
                x = wrap_sub_vec(spec.object_catas[oi], vi, self.cata_n, 4).to(dev)   # Wrap (oi, vi) vector.
                state = self.rtre.start_var(x, curr_h, dev)                     # Init var gen session.
                bitstr = [self.vocab["start"]]                                  # Current bitstring.
                bitstr_flag = True                                              # Flag indicating current bitstring continues.
                bi = 0                                                          # Bitsring iter.
                while bitstr_flag:                                              # Loop until bitstring for (oi, vi) is decided.
                    state.attempted_bits = []                                   # Bits that have been selected but cause a violation.
                    if possible[(oi, vi)][0] == possible[(oi, vi)][1]:              # Deterministic. Proceed to next var.
                        dv = possible[(oi, vi)][0]
                        print("  (%d, %d) is deterministic (%d)." % (oi, vi, dv))
                        state = self._handle_deterministic(bitstr, dv, state, dev)
                        curr_h = state.h
                        break
                    while True:                                                 # Loop through attempts for bi.
                        state_init = state.clone()
                        if len(bitstr) < 30:# and possible[(oi, vi)][0] != possible[(oi, vi)][1]:
                            b, state = self.rtre.query_bit(state, dev)
                        else:
                            b = self.vocab["end"]
                        u_table = copy.deepcopy(possible)
                        new_min, new_max = bitstring_to_number_partial(self.vocab, bitstr + [b])
                        state.attempted_bits.append(b)
                        print("  (%d, %d) bit %d -- attempted %d." % (oi, vi, bi, b))
                        if not ranges_intersect(tuple(u_table[(oi, vi)]), (new_min, new_max)):  # Check that bitstr does not violate current poss table.
                            print("    type 1 violation.")
                            state_init.reconcile(state)
                            state = state_init
                            if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                                raise UnsatError("No solution.")
                        else:
                            #print("  (%d, %d) bit %d -- attempting %d. Old range = %s. New range = %s." % (oi, vi, bi, b, u_table[(oi, vi)], [max(u_table[(oi, vi)][0], new_min), min(u_table[(oi, vi)][1], new_max)]))
                            u_table[(oi, vi)] = [max(u_table[(oi, vi)][0], new_min), min(u_table[(oi, vi)][1], new_max)]
                            u_table, viol_con = update_poss_table(u_table, spec, ledger)
                            if viol_con is not None:
                                print("    type 2 violation.")
                                #print("  ", viol_str)
                                state_init.reconcile(state)
                                state = state_init
                                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                                    print(oi, vi, bi)
                                    print(state.attempted_bits)
                                    print(possible)
                                    raise UnsatError("No solution.")
                            else:
                                #print("  (%d, %d)  bit %d -- attempted %d. accepted." % (oi, vi, bi, b))
                                print("    accepted. %s." % (u_table[(oi, vi)],))
                                bitstr.append(b)
                                if b == self.vocab["end"]:
                                    bitstr_flag = False
                                    print("  (%d, %d) bitstring:  %s." % (oi, vi, "".join(map(str, bitstr))))
                                    print("  value:  %d." % bitstring_to_number(self.vocab, bitstr))
                                possible = u_table
                                break
                    bi += 1
                    curr_h = state.h
        return (poss_table_to_res_table(possible, spec), ledger)





class DFS_PCO(RTRE_PCO):
    def __init__(self, rtre, vocab, cata_n):
        super(DFS_PCO, self).__init__()
        self.vocab = vocab
        self.rtre = rtre
        self.cata_n = cata_n

    def _handle_deterministic(self, curr_bitstr, val, state, dev):
        final_bitstr = number_to_bitstring(self.vocab, val)
        so_far = len(curr_bitstr)
        if so_far >= len(final_bitstr):
            return state
        else:
            assert curr_bitstr == final_bitstr[:so_far]
            return self.rtre.roll_forward(state, final_bitstr[so_far:], dev)

    def calculate(self, img, bounds, spec, ledger, dev, trials_n = 1):
        possible = poss_table(spec, bounds)
        possible, viol_con = update_poss_table(possible, spec, ledger)  # Update for deterministic constraints / bounds.
        print("After initial update:")
        print(possible)
        oi = 0
        curr_h = self.rtre.encode_bg(img)                                            # Init hidden vector with encoded bg image.
        while oi < len(spec.object_list):                                       # Iterate through objects.
            vi = 0
            while vi < 4:                                                       # Iterate through variables [x,y,w,h].
                x = wrap_sub_vec(spec.object_catas[oi], vi, self.cata_n, 4).to(dev)   # Wrap (oi, vi) vector.
                state = self.rtre.start_var(x, curr_h, dev)                     # Init var gen session.
                bitstr = [self.vocab["start"]]                                  # Current bitstring.
                r = self.recurse(oi, vi, 0, state, possible, bitstr, spec, ledger, dev)
                if r is None:
                    print(possible)
                    raise UnsatError("No solution.")
                else:
                    state, possible, bitstr = r
                    curr_h = state.h
                    vi += 1
            oi += 1
        return (poss_table_to_res_table(possible, spec), ledger)

    def recurse(self, oi, vi, bi, state, possible, bitstr, spec, ledger, dev):
        state.attempted_bits = []
        if possible[(oi, vi)][0] == possible[(oi, vi)][1]:
            dv = possible[(oi, vi)][0]
            print("  (%d, %d) is deterministic (%d)." % (oi, vi, dv))
            state = self._handle_deterministic(bitstr, dv, state, dev)
            return (state, possible, bitstr)
        while True:
            state_init = state.clone()
            if len(bitstr) < 30:
                print("   (%d, %d, %d)::" % (oi, vi, bi), state.attempted_bits)
                b, state = self.rtre.query_bit(state, dev)
            else:
                b = self.vocab["end"]
            u_table = copy.deepcopy(possible)
            new_min, new_max = bitstring_to_number_partial(self.vocab, bitstr + [b])
            state.attempted_bits.append(b)
            print("  (%d, %d) bit %d -- attempted %d." % (oi, vi, bi, b))
            if not ranges_intersect(tuple(u_table[(oi, vi)]), (new_min, new_max)):
                print("    type 1 violation.")
                state_init.reconcile(state)
                state = state_init
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            u_table[(oi, vi)] = [max(u_table[(oi, vi)][0], new_min), min(u_table[(oi, vi)][1], new_max)]
            viol_con = check_poss_table(u_table, spec, ledger)
            if viol_con is not None:
                print("    type 2 violation.")
                print("    violated %s." % str(viol_con))
                state_init.reconcile(state)
                state = state_init
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            print("    accepted. %s." % (u_table[(oi, vi)],))
            bitstr.append(b)
            if b == self.vocab["end"]:
                print("  (%d, %d) bitstring:  %s." % (oi, vi, "".join(map(str, bitstr))))
                print("  value:  %d." % bitstring_to_number(self.vocab, bitstr))
            r = self.recurse(oi, vi, bi + 1, state, u_table, bitstr, spec, ledger, dev)
            if r is None:
                print("    type 3 violation.")
                state_init.reconcile(state)
                state = state_init
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            return r











class DFS_PCO_V2(RTRE_PCO):
    def __init__(self, rtre, vocab, cata_n):
        super(DFS_PCO_V2, self).__init__()
        self.vocab = vocab
        self.rtre = rtre
        self.cata_n = cata_n

    def _handle_deterministic(self, curr_bitstr, val, state, dev):
        final_bitstr = number_to_bitstring(self.vocab, val)
        so_far = len(curr_bitstr)
        if so_far >= len(final_bitstr):
            return state
        else:
            assert curr_bitstr == final_bitstr[:so_far]
            return self.rtre.roll_forward(state, final_bitstr[so_far:], dev)

    def calculate(self, img, bounds, spec, ledger, dev, trials_n = 1):
        possible = poss_table(spec, bounds)
        possible, viol_con = update_poss_table(possible, spec, ledger)  # Update for deterministic constraints / bounds.
        print("After initial update:")
        print(possible)
        oi, vi = 0, 0
        bitstr = [self.vocab["start"]]
        init_h = self.rtre.encode_bg(img)                                       # Init hidden vector with encoded bg image.
        r = self.recurse(0, 0, 0, None, possible, bitstr, spec, ledger, dev, init_h = init_h)
        if r is None:
            print(possible)
            raise UnsatError("No solution.")
        else:
            state, possible, bitstr = r
        return (poss_table_to_res_table(possible, spec), ledger)

    def recurse(self, oi, vi, bi, state, possible, bitstr, spec, ledger, dev, init_h = None):
        if oi >= len(spec.object_list):
            return (state, possible, bitstr)
        if bitstr[-1] == self.vocab["start"]:
            x = wrap_sub_vec(spec.object_catas[oi], vi, self.cata_n, 4).to(dev) # Wrap (oi, vi) vector.
            if state is None:
                state = self.rtre.start_var(x, init_h, dev)                     # Init var gen session.
            else:
                state = self.rtre.start_var(x, state.h, dev)
        state.attempted_bits = []
        if possible[(oi, vi)][0] == possible[(oi, vi)][1]:
            dv = possible[(oi, vi)][0]
            print("  (%d, %d) is deterministic (%d)." % (oi, vi, dv))
            state = self._handle_deterministic(bitstr, dv, state, dev)
            vi += 1
            bitstr = [self.vocab["start"]]
            if vi >= 4:
                vi = 0
                oi += 1
                print((oi, vi))
            return self.recurse(oi, vi, 0, state.clone(), copy.deepcopy(possible), list(bitstr), spec, ledger, dev)
        while True:
            state_init = state.clone()
            if len(bitstr) < 30:
                #print("   (%d, %d, %d)::" % (oi, vi, bi), state.attempted_bits)
                b, state = self.rtre.query_bit(state, dev)
            else:
                if bitstr[-1] == self.vocab["end"]:
                    input()
                    return None
                b = self.vocab["end"]
            u_table = copy.deepcopy(possible)
            new_min, new_max = bitstring_to_number_partial(self.vocab, bitstr + [b])
            state.attempted_bits.append(b)
            print("  (%d, %d) bit %d -- attempted %d (%s)." % (oi, vi, bi, b, str(state.attempted_bits)))
            if not ranges_intersect(tuple(u_table[(oi, vi)]), (new_min, new_max)):
                #print("    type 1 violation.")
                state_init.reconcile(state)
                state = state_init
                if bitstring_to_number(self.vocab, bitstr + [self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["right"], self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["left"], self.vocab["end"]]):
                   return None
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            u_table[(oi, vi)] = [max(u_table[(oi, vi)][0], new_min), min(u_table[(oi, vi)][1], new_max)]
            viol_con = check_poss_table(u_table, spec, ledger)
            if viol_con is not None:
                #print("    type 2 violation.")
                #print("    violated %s." % str(viol_con))
                state_init.reconcile(state)
                state = state_init
                if bitstring_to_number(self.vocab, bitstr + [self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["right"], self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["left"], self.vocab["end"]]):
                   return None
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            #print("    accepted. %s." % (u_table[(oi, vi)],))
            bitstr.append(b)
            if b == self.vocab["end"]:
                print("  (%d, %d) bitstring:  %s." % (oi, vi, "".join(map(str, bitstr))))
                print("  value:  %d." % bitstring_to_number(self.vocab, bitstr))
            r = self.recurse(oi, vi, bi + 1, state.clone(), copy.deepcopy(u_table), list(bitstr), spec, ledger, dev)
            if r is None:
                #print("    type 3 violation.")
                state_init.reconcile(state)
                state = state_init
                if bitstring_to_number(self.vocab, bitstr + [self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["right"], self.vocab["end"]]) == \
                   bitstring_to_number(self.vocab, bitstr + [self.vocab["left"], self.vocab["end"]]):
                   return None
                if set(state.attempted_bits) == {self.vocab["left"], self.vocab["right"], self.vocab["end"]}:
                    return None
                continue
            return r












def print_update(oi, vi, orig, new_min, new_max, reason):
    if tuple(orig) != (new_min, new_max):
        print("Update on (%d, %d):   %s ===> (%d, %d). %s" % (oi, vi, str(orig), new_min, new_max, reason))





class RandomS1:
    def start_var(self, x, h, dev):
        return State(None, None, None, None, [])

    def query_bit(self, state, dev):
        _, _, _, _, attempted_bits = state.unpack()
        remaining = list({0, 1, 2} - set(attempted_bits))
        assert len(remaining) > 0, "Error in RandomS1.query_bit."
        return (random.choice(remaining), state)

    def encode_bg(self, img):
        return None


class RandomS1NoStop:
    def start_var(self, x, h, dev):
        return State(None, None, None, None, [])

    def query_bit(self, state, dev):
        _, _, _, _, attempted_bits = state.unpack()
        remaining = list({0, 1, 2} - set(attempted_bits))
        assert len(remaining) > 0, "Error in RandomS1.query_bit."
        remaining = list({0, 1} - set(attempted_bits))
        if len(remaining) <= 0:
            return (2, state)
        return (random.choice(remaining), state)

    def encode_bg(self, img):
        return None



def main():
    print("Testing RTRE positional constraint optimization.")
    vocab = {"left": 0, "right": 1, "end": 2, "start": 3, "pad": 4}
    vocab_inverted = {v: k for k, v in vocab.items()}
    s1 = RandomS1NoStop()
    ts = Table_PCO(s1, vocab, 8)
    print("Using Table_PCO.")
    cons = [con_wider_val(0, (96 / 640) * 1000, 0),
            con_narrower_val(0, (256 / 640) * 1000, 0),
            con_taller_val(0, (96 / 400) * 1000, 0),
            con_shorter_val(0, (256 / 400) * 1000, 0),
            con_wider_val(1, (96 / 640) * 1000, 0),
            con_narrower_val(1, (256 / 640) * 1000, 0),
            con_taller_val(1, (96 / 400) * 1000, 0),
            con_shorter_val(1, (256 / 400) * 1000, 0),
            con_wider_val(2, (96 / 640) * 1000, 0),
            con_narrower_val(2, (256 / 640) * 1000, 0),
            con_taller_val(2, (96 / 400) * 1000, 0),
            con_shorter_val(2, (256 / 400) * 1000, 0),
            right_bound(0, 1000),
            down_bound(0, 1000),
            right_bound(1, 1000),
            down_bound(1, 1000),
            right_bound(2, 1000),
            down_bound(2, 1000),
            con_left(0, 1, 100),
            con_right(1, 2, 200),
            ConstraintT2("gt", 0, 3, 1, 2, 50),]# NOTE: the cow is taller than the dog is wide.
    catas = [4, 1, 2]
    spec = Spec(["A cow", "A dog", "A horse"], cons, catas)
    res = ts.calculate(None, ((0, 1000), (0, 1000), (0, 1000), (0, 1000)), spec, None, trials_n = 3)
    print("Results:")
    print(res)
    print("Done.")



if __name__ == '__main__':
    main()

#===============================================================================
