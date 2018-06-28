import pyrosetta

from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

def filter_clash_minimize(pose, hits, clash_cutoff = 35.0, sfx = None, mmap = None):
    if (not sfx):
        sfx = ScoreFunctionFactory.create_score_function("beta_nov16")

        sfx.set_weight(scoring.hbond_bb_sc, 2.0)
        sfx.set_weight(scoring.hbond_sc, 2.0)

    if (not mmap):
        mmap = MoveMap()

        mmap.set_bb(False)
        mmap.set_chi(False)
        mmap.set_jump(1, True)

    minmov = MinMover(mmap, sfx, "dfpmin_armijo_nonmonotone", 0.01, False)

    for hit in hits:
        proto_pose = pose.clone()
        proto_pose.append_pose_by_jump(hit, len(proto_pose.residues))

        sfx(proto_pose)

        fa_rep = proto_pose.energies().total_energies()[scoring.fa_rep]
        fa_elec = proto_pose.energies().total_energies()[scoring.fa_elec]

        if (fa_rep + fa_elec > clash_cutoff):
            continue

        minmov.apply(proto_pose)

        # NOTE(onalant): Which one is the ideal solution here?
        yield proto_pose #.split_by_chain(proto_pose.num_chains())

