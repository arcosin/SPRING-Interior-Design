
import copy

from constraint_language_v2 import *

vname_map = ['x', 'y', 'w', 'h']
cname_map = {"lt": "less than", "gt": "greater than", "eq": "equals"}


def ranges_intersect(a, b):
    inter = (max(a[0], b[0]), min(a[1], b[1]))
    return (inter[0] <= inter[1])


def ranges_intersect3(a, b, c):
    inter = (max(a[0], b[0], c[0]), min(a[1], b[1], c[1]))
    return (inter[0] <= inter[1])


def cen(xy, wh):
    return xy + (wh // 2)


def range_mind(r1, r2):
  a, b = sorted((r1, r2))
  if a[0] <= a[1] < b[0]:
    return b[0] - a[1]
  return 0

def range_maxd(r1, r2):
  a, b = sorted((r1, r2))
  return b[1] - a[0]


def poss_table(spec, bounds):
    possible = {(oi, vi): [bounds[vi][0], bounds[vi][1]] for vi in range(4) for oi in range(len(spec.object_list))}
    return possible



def poss_table_to_res_table(possible, spec):
    result = dict()
    for oi, op in enumerate(spec.object_list):
        result[oi] = dict()
        for vi in range(4):
            assert possible[(oi, vi)][0] == possible[(oi, vi)][1]
            result[oi][vi] = possible[(oi, vi)][0]
    return result



def update_poss_table(possible, spec, ledger):                               # Return table if not a violation. None otherwise.
    at_start = None
    ucf = False                                                                 # Unknown constraint flag.
    while possible != at_start:
        #print("doot")
        safe = True
        at_start = copy.deepcopy(possible)
        for con in spec.constraint_list:                                        # Check for violation.
            violation_con = con
            if isinstance(con, ConstraintT2):
                c, o1, v1, o2, v2, offset = con
                if c == "lt":                                                   # o1v1 min < o2v2 max
                    point = possible[(o2, v2)][1] - offset
                    safe = (possible[(o1, v1)][0] <= point)
                    nv = min(possible[(o1, v1)][1], point)
                    possible[(o1, v1)][1] = nv
                elif c == "gt":
                    point = possible[(o2, v2)][0] + offset
                    safe = (possible[(o1, v1)][1] >= point)
                    nv = max(possible[(o1, v1)][0], point)
                    possible[(o1, v1)][0] = nv
                elif c == "eq":
                    safe = ranges_intersect(possible[(o1, v1)], possible[(o2, v2)])
                    minv = max(possible[(o1, v1)][0], possible[(o2, v2)][0])
                    maxv = min(possible[(o1, v1)][1], possible[(o2, v2)][1])
                    possible[(o1, v1)][0] = minv
                    possible[(o2, v2)][0] = minv
                    possible[(o1, v1)][1] = maxv
                    possible[(o2, v2)][1] = maxv
                else:
                    ucf = True
            elif isinstance(con, ConstraintT1):
                c, o1, v1, val, offset = con
                if c == "lt":
                    point = val - offset
                    safe = (possible[(o1, v1)][0] <= point)
                    nv = min(possible[(o1, v1)][1], point)
                    possible[(o1, v1)][1] = nv
                elif c == "gt":
                    point = val + offset
                    safe = (possible[(o1, v1)][1] >= point)
                    nv = max(possible[(o1, v1)][0], point)
                    possible[(o1, v1)][0] = nv
                elif c == "eq":
                    safe = ranges_intersect(possible[(o1, v1)], [val, val])
                    possible[(o1, v1)][0] = val
                    possible[(o1, v1)][1] = val
                else:
                    ucf = True
            elif isinstance(con, ConstraintT3):
                c, a, o1, v1, o2, v2, val, offset = con
                if c == "lt":
                    point = val - offset
                    if a == "+":
                        safe = (possible[(o1, v1)][0] + possible[(o2, v2)][0] <= point)
                        possible[(o1, v1)][1] = min(point - possible[(o2, v2)][0], possible[(o1, v1)][1])
                        possible[(o2, v2)][1] = min(point - possible[(o1, v1)][0], possible[(o2, v2)][1])
                    else:
                        ucf = True
                elif c == "gt":
                    point = val + offset
                    if a == "+":
                        safe = (possible[(o1, v1)][1] + possible[(o2, v2)][1] >= point)
                        possible[(o1, v1)][0] = max(point - possible[(o2, v2)][1], possible[(o1, v1)][0])
                        possible[(o2, v2)][0] = max(point - possible[(o1, v1)][1], possible[(o2, v2)][0])
                    else:
                        ucf = True
                elif c == "eq":
                    safe = ranges_intersect(agg, [val, val])
                else:
                    ucf = True
            elif isinstance(con, ConstraintT4):
                c, a, o1, v1, o2, v2, o3, v3, offset = con
                if c == "lt":
                    point = possible[(o3, v3)][1] - offset
                    if a == "+":
                        safe = (possible[(o1, v1)][0] + possible[(o2, v2)][0] <= point)
                        #possible[(o3, v3)][0] = max(possible[(o1, v1)][0] + possible[(o2, v2)][0] + offset, possible[(o3, v3)][0])
                        #possible[(o1, v1)][1] = min(possible[(o3, v3)][1] - possible[(o2, v2)][0] - offset, possible[(o1, v1)][1])
                        #possible[(o2, v2)][1] = min(possible[(o3, v3)][1] - possible[(o1, v1)][0] - offset, possible[(o2, v2)][1])
                    else:
                        ucf = True
                elif c == "gt":
                    point = possible[(o3, v3)][0] + offset
                    if a == "+":
                        safe = (possible[(o1, v1)][1] + possible[(o2, v2)][1] >= point)
                    else:
                        ucf = True
                elif c == "eq":
                    '''
                    if a == "+":
                        agg = [possible[(o1, v1)][0] + possible[(o2, v2)][0], possible[(o1, v1)][1] + possible[(o2, v2)][1]]
                        safe = ranges_intersect(agg, possible[(o3, v3)])
                    else:
                        ucf = True
                    '''
                else:
                    ucf = True
            else:
                ucf = True
            if not safe:
                return (possible, violation_con)
    return (possible, None)









def check_poss_table(possible, spec, ledger):
    safe, star = check_cons(possible, spec.constraint_list, m = "and")
    if not safe:
        return star
    else:
        return None








def check_cons(possible, con_list, m = "and"):
    safe = True
    for con in con_list:
        star_con = con
        safe = check_con(possible, con)
        if m == "and" and not safe:
            return (False, star_con)
        elif m == "or" and safe:
            return (True, star_con)
    if m == "and":
        return (True, None)
    elif m == "or":
        return (False, None)





def check_con(possible, con):
    safe = True
    neg_safe = True
    if isinstance(con, ConstraintT2):
        c, o1, v1, o2, v2, offset = con
        if c == "lt":
            point = possible[(o2, v2)][1] - offset
            safe = (possible[(o1, v1)][0] <= point)
            neg_safe = (possible[(o1, v1)][1] > point)
        elif c == "gt":
            point = possible[(o2, v2)][0] + offset
            safe = (possible[(o1, v1)][1] >= point)
            neg_safe = (possible[(o1, v1)][0] < point)
        elif c == "eq":
            safe = ranges_intersect(possible[(o1, v1)], possible[(o2, v2)])
            neg_safe = (possible[(o1, v1)][0] != possible[(o1, v1)][1]) or (possible[(o1, v1)] != possible[(o2, v2)])
        else:
            raise NotImplementedError()
    elif isinstance(con, ConstraintT1):
        c, o1, v1, val, offset = con
        if c == "lt":
            point = val - offset
            safe = (possible[(o1, v1)][0] <= point)
            neg_safe = (possible[(o1, v1)][1] > point)
        elif c == "gt":
            point = val + offset
            safe = (possible[(o1, v1)][1] >= point)
            neg_safe = (possible[(o1, v1)][0] < point)
        elif c == "eq":
            safe = ranges_intersect(possible[(o1, v1)], [val, val])
            neg_safe = possible[(o1, v1)] != [val, val]
        else:
            raise NotImplementedError()
    elif isinstance(con, ConstraintT3):
        c, a, o1, v1, o2, v2, val, offset = con
        if c == "lt":
            point = val - offset
            if a == "+":
                safe = (possible[(o1, v1)][0] + possible[(o2, v2)][0] <= point)
                neg_safe = (possible[(o1, v1)][1] + possible[(o2, v2)][1] > point)
            else:
                raise NotImplementedError()
        elif c == "gt":
            point = val + offset
            if a == "+":
                safe = (possible[(o1, v1)][1] + possible[(o2, v2)][1] >= point)
                neg_safe = (possible[(o1, v1)][0] + possible[(o2, v2)][0] < point)
            else:
                raise NotImplementedError()
        elif c == "eq":
            if a == "+":
                agg = [possible[(o1, v1)][0] + possible[(o2, v2)][0], possible[(o1, v1)][1] + possible[(o2, v2)][1]]
                safe = ranges_intersect(agg, [val, val])
                neg_safe = agg != [val, val]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(con, ConstraintT4):
        c, a, o1, v1, o2, v2, o3, v3, offset = con
        if c == "lt":
            point = possible[(o3, v3)][1] - offset
            if a == "+":
                safe = (possible[(o1, v1)][0] + possible[(o2, v2)][0] <= point)
            else:
                raise NotImplementedError()
        elif c == "gt":
            point = possible[(o3, v3)][0] + offset
            if a == "+":
                safe = (possible[(o1, v1)][1] + possible[(o2, v2)][1] >= point)
            else:
                raise NotImplementedError()
        elif c == "eq":
            if a == "+":
                agg = [possible[(o1, v1)][0] + possible[(o2, v2)][0], possible[(o1, v1)][1] + possible[(o2, v2)][1]]
                safe = ranges_intersect(agg, possible[(o3, v3)])
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
    elif isinstance(con, ConstraintT6):
        c, o1, o2, o3, offset = con
        if c == "mdisteq":
            cen1x = [cen(possible[(o1, 0)][0], possible[(o1, 2)][0]), cen(possible[(o1, 0)][1], possible[(o1, 2)][1])]
            cen2x = [cen(possible[(o2, 0)][0], possible[(o2, 2)][0]), cen(possible[(o2, 0)][1], possible[(o2, 2)][1])]
            cen3x = [cen(possible[(o3, 0)][0], possible[(o3, 2)][0]), cen(possible[(o3, 0)][1], possible[(o3, 2)][1])]
            cen1y = [cen(possible[(o1, 1)][0], possible[(o1, 3)][0]), cen(possible[(o1, 1)][1], possible[(o1, 3)][1])]
            cen2y = [cen(possible[(o2, 1)][0], possible[(o2, 3)][0]), cen(possible[(o2, 1)][1], possible[(o2, 3)][1])]
            cen3y = [cen(possible[(o3, 1)][0], possible[(o3, 3)][0]), cen(possible[(o3, 1)][1], possible[(o3, 3)][1])]
            mdist12 = [range_mind(cen1x, cen2x) + range_mind(cen1y, cen2y), range_maxd(cen1x, cen2x) + range_maxd(cen1y, cen2y)]
            mdist13 = [range_mind(cen1x, cen3x) + range_mind(cen1y, cen3y), range_maxd(cen1x, cen3x) + range_maxd(cen1y, cen3y)]
            mdist23 = [range_mind(cen2x, cen3x) + range_mind(cen2y, cen3y), range_maxd(cen2x, cen3x) + range_maxd(cen2y, cen3y)]
            #print(o1, possible[(o1, 0)], possible[(o1, 1)], possible[(o1, 2)], possible[(o1, 3)])
            #print(o2, possible[(o2, 0)], possible[(o2, 1)], possible[(o2, 2)], possible[(o2, 3)])
            #print(o3, possible[(o3, 0)], possible[(o3, 1)], possible[(o3, 2)], possible[(o3, 3)])
            #print(mdist12)
            #print(mdist13)
            #print(mdist23)
            safe = ranges_intersect3(mdist12, mdist13, mdist23)
        else:
            raise NotImplementedError()
    elif isinstance(con, ConstraintNOT):
        not_con = con.c
        safe = not check_con(possible, not_con)
    elif isinstance(con, ConstraintAND):
        and_cons = con.c
        safe, _ = check_cons(possible, and_cons, m = "and")
    elif isinstance(con, ConstraintOR):
        or_cons = con.c
        safe, _ = check_cons(possible, or_cons, m = "or")
    else:
        raise NotImplementedError()
    return safe



#===============================================================================
