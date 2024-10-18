import matplotlib.pyplot as plt
import numpy as np


def mitsuta(angles):
    if len(angles[~np.isnan(angles)]) == 0:
        return np.empty(angles.shape)
    else:
        angles = angles[~np.isnan(angles)]

    D = angles[0]
    Dlist = [D]
    for ang in angles[1:]:
        delta = ang - D
        if delta < -180:
            Dlist.append(D + delta + 360)
        elif abs(delta) < 180:
            Dlist.append(D + delta) # = ang
        elif delta > 180:
            Dlist.append(D + delta - 360)
        else:
            #print('* D undefined for delta = 180.')
            #print('adding .01 deg to the difference.')
            delta += 0.01
            Dlist.append(D + delta - 360)
    return Dlist


def angrange(angles):
    Dlist = mitsuta(angles)
    return np.max(Dlist)-np.min(Dlist)


def angstd(angles):
    Dlist = mitsuta(angles)

    return np.std(Dlist)

def maxdist(angles):
    Dlist = mitsuta(angles)

    diff = np.diff(np.sort(Dlist))

    if len(diff) > 0:
        return np.max(diff)
    else:
        return 0

def angavrg(angles):

    Dlist = mitsuta(angles)
    avg = np.sum(Dlist) / len(angles)

    if avg >= 360:
        return avg - 360
    elif avg < 0:
        return avg + 360
    else:
        return avg
