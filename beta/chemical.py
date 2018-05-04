from collections import namedtuple, OrderedDict

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

functional_groups = sorted([
    ("OH_", FunctionalGroup("hydroxide", True, True, ["CV", "OH", "HH"])),
    ("G__", FunctionalGroup("guanidinium", True, False, ["CZ", "NH1", "NH2"])),
    ("I__", FunctionalGroup("imidazole", True, True, ["ND1", "CD2", "NE2"])),
    # imidazole tautomer
    ("ID_", FunctionalGroup("imidazole_D", True, True, ["ND1", "CD2", "NE2"])),
    ("A__", FunctionalGroup("amine", True, False, ["NZ", "1HZ", "2HZ"])),
    ("C__", FunctionalGroup("carboxylate", False, True, ["CD", "OE1", "OE2"])),
    ("CA_", FunctionalGroup("carboxamide", True, True, ["CG", "OD1", "ND2"]))
]), key=lambda x, x[0])

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

