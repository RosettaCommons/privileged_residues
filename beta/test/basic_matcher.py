import pyrosetta

import argparse
import os
import pickle
import privileged_residues

# WARN(onalant): hack to make beta files available for import
import sys
sys.path.append("../")

from os import makedirs, path
from privileged_residues.privileged_residues import _init_pyrosetta as init
from privileged_residues import bidentify
from privileged_residues import position_residue
from process import PrivilegedResidues

def make_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("PDBFile", type=str, help="pose to match against")
    
    parser.add_argument("--cart-resl", dest="cart_resl", type=float, default=0.1, help="Cartesian resolution of table")
    parser.add_argument("--ori-resl", dest="ori_resl", type=float, default=2.0, help="Orientation resolution of table")
    parser.add_argument("--cart-bound", dest="cart_bound", type=float, default=16.0, help="Cartesian boundary of table")

    parser.add_argument("--outpath", dest="outpath", type=str, default=None, help="Output path")

    return parser

if __name__ == "__main__":
    parser = make_parser()

    args = parser.parse_args()

    init()
    
    presidues = PrivilegedResidues()

    p = pyrosetta.pose_from_pdb(path.expanduser(args.PDBFile))

    pairs_of_rays = bidentify.look_for_sc_sc_bidentates(p)
    first, second = pairs_of_rays[0]

    hits = list(presidues.match(first, second, groups=["sc_sc"]))
    print(len(hits))

    if args.outpath:
        out = path.expanduser(args.outpath)
        makedirs(out, exist_ok=True)

        p.dump_pdb(path.join(out, "ori.pdb"))

        for i, hit in enumerate(hits):
            hit.dump_pdb(path.join(out, "result_%04d.pdb" % (i + 1)))

