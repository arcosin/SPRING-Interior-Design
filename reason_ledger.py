



class RLedger(object):
    def __init__(self):
        super(RLedger, self).__init__()
        self.updates = []

    def add_update(self, var, original, update, reason):
        ev = {"var": var, "original": original, "update": update, "reason": reason}
        self.updates.append(ev)

    def add_violation(self, original, update, reason):
        ev = {"original": original, "update": update, "reason": reason}
        self.data["events"].append(ev)
