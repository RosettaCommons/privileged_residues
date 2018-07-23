import numpy as np
import pyrosetta

from pyrosetta.rosetta.core import scoring
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from rmsd import rmsd

def filter_clash_minimize(pose, hits, clash_cutoff = 35.0, rmsd_cutoff = 0.5, sfx = None, mmap = None, limit = 0):
    """Filter match output for clashes, then minimize the remaining
    structures against the target pose.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Target structure.
    hits : np.ndarray
        Set of functional group matches against positions in the target.
    clash_cutoff : float
        Maximum tolerated increase in score terms during clash checking.
    sfx : pyrosetta.rosetta.core.scoring.ScoreFunction, optional
        Scorefunction to use during minimization. If left as None, a
        default scorefunction is constructed.
    mmap : pyrosetta.rosetta.protocols.minimization_packing.MinMover, optional
        Movemap to use during minimization. If left as None, a default
        movemap is constructed.

    Yields
    ------
    pyrosetta.Pose
        Next target structure and matched functional group, minimized
        and in complex.
    """
    if (not sfx):
        sfx = scoring.ScoreFunctionFactory.create_score_function("beta_nov16")

        sfx.set_weight(scoring.hbond_bb_sc, 2.0)
        sfx.set_weight(scoring.hbond_sc, 2.0)

    if (not mmap):
        mmap = MoveMap()

        mmap.set_bb(False)
        mmap.set_chi(False)
        mmap.set_jump(1, True)

    minmov = MinMover(mmap, sfx, "dfpmin_armijo_nonmonotone", 0.01, False)

    for n, (hash, match) in enumerate(hits):
        proto_pose = pose.clone()
        proto_pose.append_pose_by_jump(match, len(proto_pose.residues))

        sfx(proto_pose)

        fa_rep = proto_pose.energies().total_energies()[scoring.fa_rep]
        fa_elec = proto_pose.energies().total_energies()[scoring.fa_elec]

        if (fa_rep + fa_elec > clash_cutoff):
            continue

        minmov.apply(proto_pose)
        minimized = proto_pose.split_by_chain(proto_pose.num_chains())
        
        ori_coords = np.array([atom.xyz() for res in match.residues for atom in res.atoms()])
        min_coords = np.array([atom.xyz() for res in minimized.residues for atom in res.atoms()])

        if (rmsd(ori_coords, min_coords) < rmsd_cutoff):
            yield (hash, minimized)

        if (limit and n + 1 >= limit):
            break

    return []

