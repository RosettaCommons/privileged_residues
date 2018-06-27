import re

from collections import defaultdict, namedtuple
from geometry import create_ray
from itertools import combinations, product

# the order of the keys of this dictionary
FunctionalGroup = namedtuple('FunctionalGroup', ['resName', 'donor', 'acceptor', 'atoms'])
FunctionalGroup.__doc__ = """Store hydrogen bonding information about a
functional group as well as information that can be used to position it
in three-space.

Attributes:
    resName (str): Name of the functional group.
    donor (bool): True if the functional group can be a donor in a
        hydrogen bond.
    acceptor (bool): True if the functional group can be an acceptor in
        a hydrogen bond.
    atoms (list): A list of three atom names (str) that are used to
        construct a coordinate frame to describe the position of the
        functional group in space.
"""

functional_groups = {
    "OH_": FunctionalGroup("hydroxide", True, True, ["CV", "OH", "HH"]),
    "G__": FunctionalGroup("guanidinium", True, False, ["CZ", "NH1", "NH2"]),
    "I__": FunctionalGroup("imidazole", True, True, ["ND1", "CD2", "NE2"]),
    # imidazole tautomer
    "ID_": FunctionalGroup("imidazole_D", True, True, ["ND1", "CD2", "NE2"]),
    "A__": FunctionalGroup("amine", True, False, ["NZ", "1HZ", "2HZ"]),
    "C__": FunctionalGroup("carboxylate", False, True, ["CD", "OE1", "OE2"]),
    "CA_": FunctionalGroup("carboxamide", True, True, ["CG", "OD1", "ND2"])
}

ResInfo = namedtuple('ResName', ['grp', 'atoms'])
rsd_to_fxnl_grp = {
    "ASN": ResInfo("CA_", ["CG", "OD1", "ND2"]),
    "GLN": ResInfo("CA_", ["CD", "OE1", "NE2"]),
    "ASP": ResInfo("C__", ["CG", "OD1", "OD2"]),
    "GLU": ResInfo("C__", ["CD", "OE1", "OE2"]),
    # Use HD1/HE2 to determine which tautomer to use
    "HIS": ResInfo("I__", ["ND1", "CD2", "NE2"]),
    "ARG": ResInfo("G__", ["CZ", "NH1", "NH2"]),
    "SER": ResInfo("OH_", ["CB", "OG", "HG"]),
    "THR": ResInfo("OH_", ["CB", "OG1", "HG1"]),
    "TYR": ResInfo("OH_", ["CZ", "OH", "HH"]),
    "LYS": ResInfo("A__", ["NZ", "1HZ", "2HZ"])
}

def _n_rays(pose, selected):
    rays = { }

    for i in range(1, len(pose.residues) + 1):
        if (selected[i]):
            rsd = pose.residue(i)

            N = rsd.atom_index("N")
            H = rsd.attached_H_begin(N)
            
            rays[i] = create_ray(rsd.xyz(H), rsd.xyz(N))

    return rays

def _c_rays(pose, selected):
    rays = { }

    for i in range(1, len(pose.residues) + 1):
        if (selected[i]):
            rsd = pose.residue(i)

            C = rsd.atom_index("C")
            O = rsd.atom_index("O")
            
            rays[i] = create_ray(rsd.xyz(O), rsd.xyz(C))

    return rays

def _sc_donor(pose, selected):
    rays = defaultdict(list)

    reg = re.compile(r"[A-Za-z]")

    for i in range(1, len(pose.residues) + 1):
        if (selected[i]):
            rsd = pose.residue(i)

            for j in range(rsd.first_sidechain_atom(), rsd.natoms() + 1):
                name = reg.search(rsd.atom_name(j)).group(0)
                hatm = rsd.attached_H_begin(j)

                if (name == "N" and hatm <= rsd.natoms()):
                    rays[i].append(create_ray(rsd.xyz(hatm), rsd.xyz(j)))

    return dict(rays)

def _sc_acceptor(pose, selected):
    rays = defaultdict(list)

    reg = re.compile(r"[A-Za-z]")

    for i in range(1, len(pose.residues) + 1):
        if (selected[i]):
            rsd = pose.residue(i)

            for j in range(rsd.first_sidechain_atom(), rsd.natoms() + 1):
                name = reg.search(rsd.atom_name(j)).group(0)
                batm = rsd.atom_base(j)

                if (name == "O"):
                    rays[i].append(create_ray(rsd.xyz(j), rsd.xyz(batm)))

    return dict(rays)

def sc_bb_rays(pose, selector):
    selected = selector.apply(pose)

    nrays = _n_rays(pose, selected)
    crays = _c_rays(pose, selected)

    rays = []

    for i in nrays.keys():
        if (i - 1 in crays):
            rays.append((nrays[i], crays[i - 1]))

        rays.append((nrays[i], crays[i]))

    return rays

def sc_scbb_rays(pose, selector):
    selected = selector.apply(pose)

    nrays = _n_rays(pose, selected)
    crays = _c_rays(pose, selected)

    sc_acc = _sc_acceptor(pose, selector)
    sc_don = _sc_donor(pose, selector)

    rays = []

    for i in nrays.keys():
        if (i in sc_acc):
            for jray in sc_acc[i]:
                rays.append((nrays[i], jray))
        if (i in sc_don):
            for jray in sc_don[i]:
                rays.append((jray, crays[i]))

    return rays

def sc_sc_rays(pose, selector):
    selected = selector.apply(pose)

    sc_acc = _sc_acceptor(pose, selected)
    sc_don = _sc_donor(pose, selected)

    rays = []

    for i in [x for x in range(1, len(pose) + 1) if x in sc_don and x in sc_acc]:
        for (jray, kray) in product(sc_don[i], sc_acc[i]):
            rays.append((jray, kray))

    for i in sc_don:
        for (jray, kray) in combinations(sc_don[i], 2):
            rays.append((jray, kray))

    for i in sc_acc:
        for (jray, kray) in combinations(sc_acc[i], 2):
            rays.append((jray, kray))

    return rays

def donor_donor_rays(pose, selector):
    selected = selector.apply(pose)

    sc_don = _sc_donor(pose, selected)

    rays = []

    for (i, j) in product(sc_don.keys(), sc_don.keys()):
        if (i != j):
            for (kray, lray) in product(sc_don[i], sc_don[j]):
                rays.append((kray, lray))

    return rays

def acceptor_acceptor_rays(pose, selector):
    selected = selector.apply(pose)

    sc_acc = _sc_acceptor(pose, selected)

    rays = []

    for (i, j) in product(sc_acc.keys(), sc_acc.keys()):
        if (i != j):
            for (kray, lray) in product(sc_acc[i], sc_acc[j]):
                rays.append((kray, lray))

    return rays

def donor_acceptor_rays(pose, selector):
    selected = selector.apply(pose)

    sc_don = _sc_donor(pose, selected)

    rays = []

    for (i, j) in product(sc_don.keys(), sc_acc.keys()):
        if (i != j):
            for (kray, lray) in product(sc_don[i], sc_acc[j]):
                rays.append((kray, lray))

    return rays
