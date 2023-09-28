
class State:
    def __init__(self, h, z, tok_oh, last_out, attempted_bits):
        super(State, self).__init__()
        self.z = z
        self.h = h
        self.tok_oh = tok_oh
        self.last_out = last_out
        self.attempted_bits = attempted_bits

    def unpack(self):
        return (self.h, self.z, self.tok_oh, self.last_out, self.attempted_bits)

    def clone(self, shallow = False):
        if shallow:
            return State(self.h, self.z, self.tok_oh, self.last_out, list(self.attempted_bits))
        else:
            return State(self.h.clone(), self.z.clone(), self.tok_oh.clone(), self.last_out, list(self.attempted_bits))

    def reconcile(self, s2):
        self.last_out = s2.last_out
        self.attempted_bits = s2.attempted_bits



#===============================================================================
