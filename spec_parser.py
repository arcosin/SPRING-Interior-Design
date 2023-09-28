
import re

from spec import Spec
from constraint_language_v2 import *

example_constraints = [
"object 1 is above object 2.",
"object 1 is below object 2.",
"object 3 is right of object 0.",
"object 3 is left of object 0.",
"object 4 is aligned on y with object 1.",
"object 4 is aligned on x with object 3.",
"object 1 is above object 2 with offset 50.",
"object 1 is below object 2 with offset -30.",
"object 3 is right of object 0 with offset -20.",
"object 3 is left of object 0 with offset 1.",
]

#t1_regex = r"^[^\d]*(\d+)[^\d]*(above|below|right|left)[^\d]*(\d+)[^\d]*$"                        # Matches a, b, r, l constraints.
#t2_regex = r"^[^\d]*(\d+)[^\d]*aligned[^\d]*(x|y)[^\d]*(\d+)[^\d]*$"                              # Matches aligning constraints.
#t3_regex = r"^[^\d]*(\d+)[^\d]*(above|below|right|left)[^\d]*(\d+)[^\d]*offset[^\d-]*(-?\d+)$"    # Matches a, b, r, l constraints with offset.

t1_regex = r"^(above|below|right|left|wider|narrower|shorter|taller|xeq|yeq|weq|heq|aboveabove|belowbelow|rightright|leftleft)(?:_(value))?\((\d+),(\d+)(?:,(\d+))?\)$"
#t2_regex = r"^(type|prompt)\((\d+),(\d+)\)$"


class Scanner:
    def __init__(self, f = None):
        super(Scanner, self).__init__()
        self.f = f
        if f is not None:
            self.fi = 0
            ff = open(f, "r")
            self.data = ff.read().split("\n")
            ff.close()

    def scan(self, p = None):
        if self.f is None:
            if p is None:
                return input()
            else:
                return input(p)
        else:
            if self.fi < len(self.data):
                r = self.data[self.fi]
                self.fi += 1
                return r
            else:
                raise Exception("Error: file ran out.")





def sent_to_constraint(s):
    s = s.lower().replace(" ", "")
    t1_match = re.match(t1_regex, s)
    if t1_match is not None:
        con, is_val, a, b, offset = t1_match.groups()
        if offset is None:
            offset = 0
        try:
            if is_val is None:
                if con in ["leftleft", "rightright", "aboveabove", "belowbelow"]:
                    return t4_map[con](int(a), int(b), int(offset))
                else:
                    return t2_map[con](int(a), int(b), int(offset))
            else:
                return t1_map[con](int(a), int(b), int(offset))
        except Exception as e:
            print(e)
            print("oof")
            pass
    return None






def spec_parser(cata_dict, scan_data = [], num_objs = None, defcon_builder = None, read_file = None, defcon_bounds = None):
    i = 0
    object_list = []
    painteds = []
    object_catas = []
    paint_flags = []
    cons = []
    catas = list(cata_dict.keys())
    sc = Scanner(read_file)
    for sco in scan_data:
        print("Found scanned object %d, category %d." % (i, sco["cata"]))
        object_list.append("?")
        object_catas.append(sco["cata"])
        paint_flags.append(False)
        cons += set_loc(i, sco["box"][0], sco["box"][1], sco["box"][2], sco["box"][3])
        i += 1
    if num_objs is not None:
        print("Your model supports %d objects." % num_objs)
        obj_n = num_objs
    else:
        obj_n = int(sc.scan("How many objects should be placed:\n"))
    for o in range(obj_n):
        painteds.append(i)
        paint_flags.append(True)
        prompt = sc.scan("Input the prompt for object %d.\n" % i)
        object_list.append(prompt)
        cata = None
        while cata not in catas:
            cata = sc.scan("Input the category for object %d.\nOptions: %s.\n" % (i, str(cata_dict)))
            try:
                cata = int(cata)
            except:
                cata = None
        object_catas.append(cata)
        i += 1
    if defcon_builder is not None:
        cons += defcon_builder(painteds)#, *defcon_bounds)
        print("Default constraints added for objects: %s." % str(painteds))
    if read_file is None:
        print("Input rules one line at a time and conclude with a '?'.")
    s = ''
    while s != '?':
        if len(s) > 0:
            c = sent_to_constraint(s)
            if c is None:
                print("Constraint not recognized.")
            else:
                cons.append(c)
        s = sc.scan()
    spec = Spec(object_list, cons, object_catas=object_catas, paint_flags=paint_flags)
    print(spec)
    return spec





#===============================================================================
