


class Spec(object):
    def __init__(self, object_list, constraint_list, object_catas = None, paint_flags = None):
        super(Spec, self).__init__()
        self.object_list = object_list
        if object_catas is None:   # Assume classes are dictated by object order.
            self.object_catas = list(range(len(object_list)))
        else:
            assert len(object_catas) == len(object_list)
            self.object_catas = object_catas
        if paint_flags is None:
            self.paint_flags = [True for _ in self.object_list]
        else:
            self.paint_flags = paint_flags
        self.constraint_list = constraint_list

    def __repr__(self):
        s = ""
        s += "=====================" + "\n"
        s += "         Spec        " + "\n"
        s += "---------------------" + "\n"
        for i, o in enumerate(self.object_list):
            s += "   %d [%d])  %s." % (i, self.object_catas[i], o) + "\n"
        s += "---------------------" + "\n"
        for c in self.constraint_list:
            s += "   -  %s." % str(c) + "\n"
        s += "=====================" + "\n\n"
        return s



#===============================================================================
