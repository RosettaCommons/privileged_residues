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
data = [ (path.basename(bdtype), path.splitext(respair)[0]) for (bdtype, _, respairs) in os.walk(path.join(curdir, "data", "bidentates")) for respair in respairs ]

def idfn(value):
    return "%s - %s" % value

@pytest.mark.parametrize("bdtype, respair", data, ids=idfn)
def test_bidentate(bdtype, respair):
    pdb = path.join(curdir, "data", "bidentates", bdtype, "%s.pdb" % (respair))
    pose = next(privileged_residues.util.models_from_pdb(pdb))

    selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    pres = privileged_residues.PrivilegedResidues()

    selector.append_index(5)

    ref_res = pose.residue(2)
    ref_coords = np.stack([ref_res.atom(n).xyz() for n in chemical.rsd_to_fxnl_grp[ref_res.name()].atoms])

    min_rmsd = 1000.

    for (hash, p) in pres.search(pose, [bdtype], selector):
        res = p.residue(1)
        coords = np.stack([res.atom(n).xyz() for n in chemical.functional_groups[res.name()].atoms])

        RMSD = rmsd(ref_coords, coords)
        
        if (RMSD < min_rmsd):
            min_rmsd = RMSD

        if (RMSD < 0.25):
            return

    print("Minimum RMSD across matched structures: %.3f" % min_rmsd)
    assert(False)

