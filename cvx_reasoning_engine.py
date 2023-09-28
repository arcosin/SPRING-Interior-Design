

import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp






class CVX_Reasoning_Engine(torch.nn.Module):
    def __init__(self, z_size, num_objs, constraint_list = [], x_size = 0, bounds = None):
        super(CVX_Reasoning_Engine, self).__init__()
        self.num_objs = num_objs
        self.z_size = z_size
        self.x_size = x_size
        self.bounds = bounds
        self.curr_constraint_list = constraint_list
        self.parab_layer = ParabolicLayer(self.num_objs, constraint_list)
        self.fc1 = nn.Linear(z_size + x_size + 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_objs * 4)
        self.debug_vector = None

    def obj_dict(self):
        sd = {
            "num_objs": self.num_objs,
            "z_size": self.z_size,
            "x_size": self.x_size,
            "bounds": self.bounds,
            "curr_constraint_list": self.curr_constraint_list,
            "state_dict": self.state_dict()
        }
        return sd

    def load_obj_dict(self, obj_dict):
        self.num_objs = obj_dict["num_objs"]
        self.z_size = obj_dict["z_size"]
        self.x_size = obj_dict["x_size"]
        self.change_constraints(obj_dict["curr_constraint_list"], obj_dict["bounds"])
        self.load_state_dict(obj_dict["state_dict"])

    def change_constraints(self, constraint_list = [], bounds = None):
        self.bounds = bounds
        self.curr_constraint_list = constraint_list
        self.parab_layer = ParabolicLayer(self.num_objs, constraint_list)

    def forward(self, z, bounds = None, x = None, use_cvx = 0, mapped = True):
        bs = z.size(0)
        if bounds == None:
            if self.bounds == None:
                raise ValueError("Error: Bounds must be either passed to the method or set for the object.")
            bounds = self.bounds
        bounds = bounds.unsqueeze(0).repeat(bs, 1)
        if self.x_size > 0:
            z = torch.cat([z, x, bounds], dim = 1)
        else:
            z = torch.cat([z, bounds], dim = 1)
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = F.dropout(x, 0.3)
        x = self.fc_out(x)
        self.debug_vector = x.clone().detach()
        if use_cvx == 2:
            y = (self.parab_layer(x, bounds), torch.split(x, 1, -1))
        elif use_cvx == 1:
            y = self.parab_layer(x, bounds)
        else:   # Elif use_cvx == 0.
            y = torch.split(x, 1, -1)   # Tuple of tensors (bs, 1). Tuple size 4 * n_obj.
        return y









class CVX_Placer():
    def __init__(self, num_objs, constraint_list = []):
        super(CVX_Placer, self).__init__()
        self.num_objs = num_objs
        self.parab_layer = ParabolicLayer(self.num_objs, constraint_list)
        self.pos_out_n = self.parab_layer.pos_param_len
        self.rem_out_n = self.parab_layer.param_len - self.pos_out_n

    def get_param_info(self):
        return (self.pos_out_n, self.rem_out_n, self.parab_layer.labels)

    def get_output_info(self):
        return dict(self.parab_layer.out_map)

    def change_constraints(self, constraint_list = []):
        self.parab_layer = ParabolicLayer(self.num_objs, constraint_list)

    def __call__(self, p, bounds, mapped = True):
        if mapped:
            return self.parab_layer.run_mapped(p, bounds)
        else:
            return self.parab_layer(p, bounds)











class ParabolicLayer(torch.nn.Module):
    def __init__(self, num_objs, constraint_list):
        super(ParabolicLayer, self).__init__()
        self.num_objs = num_objs
        self.out_map = dict()
        x_vars = []
        y_vars = []
        w_vars = []
        h_vars = []
        all_vars = []
        params = []
        bounds = [cp.Parameter(1), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1)]
        cons = []
        eq_bias = cp.Constant(0)
        eq = eq_bias
        for i in range(num_objs):
            xy_parab, vx, vy, px, py, xy_term_cons = self._point_term()
            x_vars.append(vx)
            y_vars.append(vy)
            w_vars.append(cp.Variable(1))
            h_vars.append(cp.Variable(1))
            cr_parab, pw, ph, cr_term_cons = self._cross_term(w_vars[i], h_vars[i], x_vars[i], y_vars[i])
            params += [px, py, pw, ph]
            all_vars += [x_vars[i], y_vars[i], w_vars[i], h_vars[i]]
            cons += xy_term_cons + cr_term_cons
            eq = eq + xy_parab + cr_parab
            cons.append(x_vars[i] <= w_vars[i] - cp.Constant(1.0))
            cons.append(y_vars[i] <= h_vars[i] - cp.Constant(1.0))
            cons.append(x_vars[i] >= bounds[0])
            cons.append(y_vars[i] >= bounds[1])
            cons.append(w_vars[i] <= bounds[2])
            cons.append(h_vars[i] <= bounds[3])
        for con in constraint_list:
            if len(con) == 3:
                o1, c, o2 = con
                off = 1.0
            else:
                o1, c, o2, off = con
                off = round(off)
            if c == 'l':                                                        # O1 left of O2.
                cons.append(x_vars[o1] <= x_vars[o2] - cp.Constant(off))
            elif c == 'r':                                                      # O1 right of O2.
                cons.append(x_vars[o1] >= x_vars[o2] + cp.Constant(off))
            elif c == 'cx':                                                     # O1 centered on x with O2.
                cons.append(x_vars[o1] == x_vars[o2])
            elif c == 'a':                                                      # O1 above O2.
                cons.append(y_vars[o1] <= y_vars[o2] - cp.Constant(off))
            elif c == 'b':                                                      # O1 below O2.
                cons.append(y_vars[o1] >= y_vars[o2] + cp.Constant(off))
            elif c == 'cy':                                                     # O1 centered on y with O2.
                cons.append(y_vars[o1] == y_vars[o2])
            elif c == "xlt":
                cons.append(x_vars[o1] <= cp.Constant(o2))
            elif c == "xgt":
                cons.append(x_vars[o1] >= cp.Constant(o2))
            elif c == "xeq":
                cons.append(x_vars[o1] == cp.Constant(o2))
            elif c == "ylt":
                cons.append(y_vars[o1] <= cp.Constant(o2))
            elif c == "ygt":
                cons.append(y_vars[o1] >= cp.Constant(o2))
            elif c == "yeq":
                cons.append(y_vars[o1] == cp.Constant(o2))
            elif c == "wlt":
                cons.append(w_vars[o1] - x_vars[o1] <= cp.Constant(o2))
            elif c == "wgt":
                cons.append(w_vars[o1] - x_vars[o1] >= cp.Constant(o2))
            elif c == "weq":
                cons.append(w_vars[o1] - x_vars[o1] == cp.Constant(o2))
            elif c == "hlt":
                cons.append(h_vars[o1] - y_vars[o1] <= cp.Constant(o2))
            elif c == "hgt":
                cons.append(h_vars[o1] - y_vars[o1] >= cp.Constant(o2))
            elif c == "heq":
                cons.append(h_vars[o1] - y_vars[o1] == cp.Constant(o2))
        prob = cp.Problem(cp.Minimize(eq), cons)                                # Setup as maximization problem. Parameters are controlled by a downstream ANN.
        self.layer = CvxpyLayer(prob, params + bounds, all_vars)                # Put in a CvxpyLayer for backprop.
        self.param_len = len(params)
        self.pos_param_len = 0

    def _parabolic_term(self, v, b):
        temp = cp.Variable(1)
        eq = cp.square(temp)
        cons = [temp == v - b]
        return (eq, cons)

    def _point_term(self):
        px = cp.Parameter(1)
        vx = cp.Variable(1)
        parab_x, pcx = self._parabolic_term(vx, px)
        py = cp.Parameter(1)
        vy = cp.Variable(1)
        parab_y, pcy = self._parabolic_term(vy, py)
        cons = pcx + pcy
        return (parab_x + parab_y, vx, vy, px, py, cons)

    def _cross_term(self, vx1, vy1, vx2, vy2):
        px = cp.Parameter(1)
        parab_x, pcx = self._parabolic_term(vx1 - vx2, px)
        py = cp.Parameter(1)
        parab_y, pcy = self._parabolic_term(vy1 - vy2, py)
        cons = pcx + pcy
        return (parab_x + parab_y, px, py, cons)

    def forward(self, x, bounds):
        x = torch.cat([x, bounds], dim = -1)
        return self.layer(*torch.split(x, 1, dim = -1), solver_args={"max_iters": 10000})

    def run_mapped(self, x, bounds):
        y = self(x, bounds)
        return {k: (y[v1], y[v2], y[v3], y[v4]) for k, (v1, v2, v3, v4) in self.out_map.items()}








#===============================================================================
