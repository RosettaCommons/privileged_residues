import pyrosetta
import privileged_residues
from privileged_residues.privileged_residues import _init_pyrosetta as init
import numpy as np
from privileged_residues import bidentify as bd

init()

cart_resl, ori_resl, cart_bound = 0.1, 2.0, 16.0
p = pyrosetta.pose_from_file('sc_bb_example.pdb')
pairs, ht = privileged_residues.find_privileged_interactions_in_pose(p)

hits = bd.look_up_interactions(pairs, ht, cart_resl, ori_resl, cart_bound)

for i, hit in enumerate(hits):
    hit.dump_pdb('result_{:04d}.pdb'.format(i + 1))
    