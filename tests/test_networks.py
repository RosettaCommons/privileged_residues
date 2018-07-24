import numpy as np
import os
import pytest

from os import path
from rmsd import rmsd

privileged_residues = pytest.importorskip("privileged_residues")
pyrosetta = pytest.importorskip("pyrosetta")
rif = pytest.importorskip("rif")

from privileged_residues import chemical

curdir = path.dirname(path.abspath(__file__))
data = [ ("network", path.splitext(rstrplt)[0]) for rstrplt in os.listdir(path.join(curdir, "data", "networks")) ]

def idfn(value):
    return "%s - %s" % value

@pytest.mark.parametrize("nwtype, restrplt", data, ids=idfn)
def test_network(nwtype, restrplt):
    pdb = path.join(curdir, "data", "networks", "%s.pdb" % (restrplt))
    pose = next(privileged_residues.util.models_from_pdb(pdb))

    selector = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    pres = privileged_residues.PrivilegedResidues()

    ref_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("X")
    selected = ref_sel.apply(pose)
    ref_res = pose.residue(list(selected).index(True))
    ref_coords = np.stack([ref_res.atom(n).xyz() for n in chemical.functional_groups[ref_res.name()].atoms])

    min_rmsd = 1000.

    for (hash, p) in pres.search(pose, [nwtype], selector):
        res = p.residue(1)
        coords = np.stack([res.atom(n).xyz() for n in chemical.functional_groups[res.name()].atoms])

        RMSD = rmsd(ref_coords, coords)

        if (RMSD < min_rmsd):
            min_rmsd = RMSD

        if (RMSD < 0.25):
            return

    print("Minimum RMSD across matched structures: %.3f" % min_rmsd)
    assert(False)

