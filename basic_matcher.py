import pyrosetta

from pyrosetta.rosetta.core.select.residue_selector import SecondaryStructureSelector

import argparse
import os
import pickle
import privileged_residues
import sys

from os import makedirs, path
from privileged_residues.privileged_residues import _init_pyrosetta as init
from privileged_residues import bidentify
from privileged_residues import position_residue

def make_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("PDBFile", type=str, help="pose to match against")
    
    parser.add_argument("--cart-resl", dest="cart_resl", type=float, default=0.1, help="Cartesian resolution of table")
    parser.add_argument("--ori-resl", dest="ori_resl", type=float, default=2.0, help="Orientation resolution of table")
    parser.add_argument("--cart-bound", dest="cart_bound", type=float, default=16.0, help="Cartesian boundary of table")

    parser.add_argument("--outpath", dest="outpath", type=str, default=None, help="Output path")

    parser.add_argument("--residues", dest="residues", type=int, default=None, nargs="*", help="Residue indices")

    parser.add_argument("--clash-cutoff", dest="clash_cutoff", type=float, default=35., help="Tolerance for clash checking")

    parser.add_argument("--n-best", dest="n_best", type=int, default=10, help="Number of top interactions to isolate")
	parser.add_argument("--params", dest="params", type=str, nargs="*", help="Additional parameter files")

    return parser

def main(argv):
    parser = make_parser()
    args = parser.parse_args()

    init()

    params = (args.cart_resl, args.ori_resl, args.cart_bound)
    ht_name = 'Sc_Sc_%.1f_%.1f_%.1f.pkl' % params
    ht_path = path.expanduser('~weitzner/bidentate_hbond_pdbs_2/hash_tables/comb/')
    ht_name_full = path.join(ht_path, ht_name)
    table_data = position_residue._fname_to_HTD('Sc_Sc_%.1f_%.1f_%.1f.pkl' % params)

    p = pyrosetta.pose_from_pdb(path.expanduser(args.PDBFile))

    with open(ht_name_full, 'rb') as f:
        ht = pickle.load(f)

    selector = SecondaryStructureSelector("L")

    pairs_of_rays = bidentify.look_for_sc_sc_bidentates(p, selector)

    hits = list(bidentify.look_up_interactions(pairs_of_rays, ht, *params))

    if args.outpath:
        out = path.expanduser(args.outpath)
        makedirs(out, exist_ok=True)

        p.dump_pdb(path.join(out, "ori.pdb"))

        for i, hit in enumerate(hits):
            hit.dump_pdb(path.join(out, "result_%04d.pdb" % (i + 1)))


if __name__ == "__main__":
    main(sys.argv)    