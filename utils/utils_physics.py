from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from .utils_vec import *


def kinetic(vels, m=None, combine=False):
    """Get kinetoc energy
    Args
        sequences of velocities: velocities of all entities
        m: mass of each entitiy; default is 1
        combine: whether sum over all the kinetic energy of all entities
    Return
        kinetic energy
    """
    num_entities = len(vels)
    if m is None:
        m = [1.0] * num_entities
    Ts = [None] * num_entities
    min_length = None
    for entity_id, vs in enumerate(vels):
        Ts[entity_id] = [None] * len(vs)
        min_length = len(vs) if min_length is None else min(min_length, len(vs))
        for t, v in enumerate(vs):
            Ts[entity_id][t] = 0.5 * m[entity_id] * (norm(v) ** 2.0)
    if combine:
        Ts_sum = [sum([Ts[entity_id][t] for entity_id in range(num_entities)]) / num_entities
                    for t in range(min_length)]
        return Ts_sum
    else:
        return Ts


def potential_spring(pos1, pos2, equ_len, k):
    """Get potential energy (a spring)
    Args
        pos1, pos2: two ends of the spring
        equ_len: equilibrium length
        k: coefficient
    Return
        potential energy
    """
    return 0.5 * k * (dist(pos1, pos2) - equ_len) ** 2.0


def force_coulomb(charge1, charge2, pos1, pos2):
    """pos2 -> pos1"""
    K = 8.98 * 10 ** 9 / (1e6)
    r = _get_dist(pos1, pos2)
    coeff = charge1 * charge2 * K  / (r ** 2)
    return (coeff * (pos1[0] - pos2[0]), coeff * (pos1[1] - pos2[1]))


def force_spring(ref_pos, pos, equ_len, k):
    d = dist(ref_pos, pos)
    f_mag = -k * (d - equ_len)
    return ((pos[0] - ref_pos[0]) / d * f_mag, (pos[1] - ref_pos[1]) / d * f_mag)