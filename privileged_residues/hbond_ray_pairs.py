import argparse
import logging
import numpy as np
import sys

from collections import namedtuple

# The following packages are not pip-installable
# The import calls are wrapped in a try/except block
try:
    import pyrosetta
    from pyrosetta.bindings.utility import bind_method
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set
    from pyrosetta.rosetta.core.select.residue_selector import LayerSelector
    HAVE_PYROSETTA = True
except ImportError:
    print('Module "pyrosetta" not found in the current environment! '
          'Go to http://www.pyrosetta.org to download it.')
    HAVE_PYROSETTA = False
    pass

try:
    import rif
    import rif.geom.ray_hash as rh
except ImportError:
    print('Module "rif" not found in the current environment! '
          'Go to https://github.com/willsheffler/rif for more information.')
    pass

logging.basicConfig(level=logging.WARN)


if HAVE_PYROSETTA:
    @bind_method(pyrosetta.rosetta.numeric.xyzVector_double_t)
    def __iter__(self):
        """Generator for pyrosetta.rosetta.numeric.xyzVector_double_t
        instances. Makes casting directly to numpy.array possible.

        Yields:
            float: The next coordinate value in the vector in the order
                x, y, z.

        Examples:
            >>> print([i for i in pose.residue(1).xyz(1)])
            [13.092, 4.473, -2.599]

            >>> np.array([*pose.residue(1).xyz(1)])
            array([ 13.092,   4.473,  -2.599])
        """
        for value in [self.x, self.y, self.z]:
            yield value


def find_hbonds(p, derivatives=False, exclude_bb=True, exclude_bsc=True,
                exclude_scb=True, exclude_sc=False):
    """Find all hydrogen bonds of a particular type in a supplied Pose
    and return them as a HBondSet.

    Args:
        p (pyrosetta.Pose): The Pose from which to extract hydrogen
            bonds.
        derivatives (bool, optional): Evaluate energy derivatives and
            store them in the HBondSet. Defaults to False.
        exclude_bb (bool, optional): If True, do not store
            backbone--backbone hydrogen bonds in the HBondSet. Defaults
            to True.
        exclude_bsc (bool, optional): If True, do not store
            backbone--side chain hydrogen bonds in the HBondSet.
            Defaults to True.
        exclude_scb (bool, optional): If True, do not store
            side chain--backbone hydrogen bonds in the HBondSet.
            Defaults to True.
        exclude_sc (bool, optional): If True, do not store
            side chain--side chain hydrogen bonds in the HBondSet.
            Defaults to False.

    Returns:
        pyrosetta.rosetta.core.scoring.hbonds.HBondSet: A hydrogen bond
        set containing the specified types of hydrogen bonds in the
        Pose.
    """
    p.update_residue_neighbors()
    hbset = HBondSet()
    fill_hbond_set(p, derivatives, hbset, exclude_bb, exclude_bsc, exclude_scb,
                   exclude_sc)
    return hbset


def surface_only():
    """Construct a LayerSelector to select surface residues only and
    return it.

    Notes:
        The LayerSelector is configured to use side-chain neighbors to
        determine burial. The cutoff for core is set to 3.5, surface to
        2.0.

    Returns:
        pyrosetta.rosetta.core.select.residue_selector.LayerSelector:
        A selector that will return a vector of boolean values where
        True indicates a surface residue and False indicates a core
        residue.
    """
    # use this with a LayerSelector to control which residues are in the set
    select_surface = LayerSelector()
    select_surface.set_use_sc_neighbors(True)
    select_surface.set_cutoffs(3.5, 2.0)  # sc_nbr cutoffs for core & surface
    select_surface.set_layers(False, False, True)
    return select_surface


def identify_bonded_pairs(p, hbset):
    """Construct an HBondSet that only contains hydrogen bonded
    residues that are on the surface of the supplied Pose and return
    it.

    Args:
        p (pyrosetta.Pose): The Pose to be examined.
        hbset (pyrosetta.rosetta.core.scoring.hbonds.HBondSet): A
            hydrogen bond set to be subsetted based on the burial of
            each residue.

    Returns:
        pyrosetta.rosetta.core.scoring.hbonds.HBondSet: A hydrogen bond
        set containing only residues on the surface of the Pose.
    """
    return HBondSet(hbset, surface_only().apply(p))


def create_ray(center, base):
    """Create a ray of unit length from two points in space and return
    it.

    Notes:
        The ray is constructed such that:

        1. The direction points from `base` to `center` and is unit
           length.
        2. The point at which the ray is centered is at `center`.

    Args:
        center (numpy.array): A (2, 3) array representing the
            coordinate at which to center the resulting ray.
        base (numpy.array): A (2, 3) array representing the base used
            to determine the direction of the resulting ray.

    Returns:
        numpy.array: A (2,4) array representing a ray in space with a
        point and a unit direction.
    """
    direction = center - base
    direction /= np.linalg.norm(direction)

    ray = np.empty((2, 4))
    ray[0][:-1] = center
    ray[0][-1] = 1.  # point!

    ray[1][:-1] = direction
    ray[1][-1] = 0.  # direction!

    return ray


def find_ray_pair_for_residues(don_rsd, acc_rsd):
    """Create a pair of rays describing a hydrogen bond betweeen
    the supplied residues and return them.

    Notes:
        Implementation is incomplete.

    Args:
        don_rsd (pyrosetta.rosetta.core.conformation.Residue):
            The hydrogen bond donor residue.
        acc_rsd (pyrosetta.rosetta.core.conformation.Residue):
            The hydrogen bond acceptor residue.

    Returns:
        tuple: A tuple of rays represented as numpy.arrays with shape
        (2, 4) representing a point and a unit direction. The first ray
        describes the donor, the second describes the acceptor.
    """
    if True:
        for i in range(1, don_rsd.natoms() + 1):
            if don_rsd.atom_is_backbone(i):
                continue
            if don_rsd.atom_type(i).is_donor():
                print(don_rsd.heavyatom_has_polar_hydrogens(i))

            if don_rsd.atom_is_polar_hydrogen(i):
                print(i)
                print(don_rsd.atom_name(i))
                base = don_rsd.atom_base(i)
                print(don_rsd.atom_name(base))

        # print(don_rsd)
        # print(acc_rsd)
        return

    hbond_dir = hydrogen - acc_heavy
    hbond_dir /= np.linalg.norm(hbond_dir)
    # when populating the hash table, we need to figure out which orbital
    # to use for the ray. when searching the hash table, we will test all of
    # them.

    HbondOverlap = namedtuple('HbondOverlap', ['projection', 'vector'])
    candidate_dirs = list()
    for orbital_id in pose.residue(hb.acc_res()).bonded_orbitals(hb.acc_atm()):
        orb_coord = np.array(
                    [*pose.residue(hb.acc_res()).orbital_xyz(orbital_id)])
        orb_dir = orb_coord - acc_heavy
        orb_dir /= np.linalg.norm(orb_dir)
        candidate_dirs.append(HbondOverlap(np.dot(orb_dir, hbond_dir),
                                           orb_coord))
    acc_orb = max(candidate_dirs, key=lambda k: k.projection).vector
    hb_acc_ray = create_ray(acc_orb, acc_heavy)


def find_ray_pair_for_hbond(pose, hb):
    """Create a pair of rays describing a hydrogen bond and return
    them.

    Notes:
        The first ray is centered on the donated hydrogen and points
        along the donor base--hydrogen direction; the second ray is
        centered on acceptor heavy atom and points in the acceptor
        heavy atom's base atom--acceptor heavy atom direction.

    Args:
        pose (pyrosetta.Pose): The Pose from which to extract
            coordinates.
        hb (pyrosetta.rosetta.core.scoring.hbonds.HBond): The hydrogen
            bond of interest.

    Returns:
        tuple: A tuple of rays represented as numpy.arrays with shape
        (2, 4) representing a point and a unit direction. The first ray
        describes the donor, the second describes the acceptor.
    """
    # donor
    don_heavy_atm_id = pose.residue(hb.don_res()).atom_base(hb.don_hatm())
    hydrogen = np.array([*pose.residue(hb.don_res()).xyz(hb.don_hatm())])
    don_heavy = np.array([*pose.residue(hb.don_res()).xyz(don_heavy_atm_id)])

    hb_don_ray = create_ray(hydrogen, don_heavy)

    # acceptor
    acc_heavy = np.array([*pose.residue(hb.acc_res()).xyz(hb.acc_atm())])
    acc_base_atm_no = pose.residue(hb.acc_res()).atom_base(hb.acc_atm())
    acc_base = np.array([*pose.residue(hb.acc_res()).xyz(acc_base_atm_no)])

    hb_acc_ray = create_ray(acc_heavy, acc_base)

    return hb_don_ray, hb_acc_ray


def hash_rays(r1, r2, resl=2, lever=10):
    """Hash pairs of rays and return an array of hased values.

    Notes:
        `r1` and `r2` must be the same shape.

    Args:
        r1 (numpy.array): An (N, 2, 4) array representing N rays in
            space each with a point and a unit direction.
        r2 (numpy.array): An (N, 2, 4) array representing N rays in
            space each with a point and a unit direction.

    Returns:
        numpy.array: An (N,) array of the hashed values (numpy.uint64)
        for each pair of rays.
    """
    h = rh.RayToRay4dHash(resl, lever, bound=1000)

    # convert from numpy to RIF types
    def _numpy_to_rif_ray(r):
        return r.astype('f4').reshape(r.shape[:-2] + (8,)).view(rif.geom.Ray)

    keys = h.get_keys(*(_numpy_to_rif_ray(r) for r in [r1, r2]))
    return keys.squeeze()


def get_frame_for_coords(coords):
    """Construct a local coordinate frame (homogenous transform) from
    three points (0-2) in space and return it.

    Notes:
        The frame is constructed such that:

        1. The point is at p2.
        2. The z axis points along p1 to p2.
        3. The y axis is in the p0-p1-p2 plane.
        4. The x axis is the cross product of y and z.

    Args:
        coords (numpy.array): A (3, 3) array representing the cartesian
            coordinates of three points in R3.

    Returns:
        numpy.array: A (4, 4) array representing the coordinate frame
        as a homogenous transform.
    """

    assert(coords.shape == (3, 3))

    frame = np.empty((4, 4))
    frame[:3, 3] = coords[2]
    frame[3, 3] = 1.

    z_hat = coords[2] - coords[1]
    z_hat /= np.linalg.norm(z_hat)
    frame[:3, 2] = z_hat

    y_hat = coords[0] - coords[1]
    y_hat -= np.dot(y_hat, z_hat) * z_hat
    y_hat /= np.linalg.norm(y_hat)
    assert(np.isclose(np.dot(y_hat, z_hat), 0.))
    frame[:3, 1] = y_hat

    x_hat = np.cross(y_hat, z_hat)
    assert(np.isclose(np.linalg.norm(x_hat), 1.))
    frame[:3, 0] = x_hat
    assert(np.isclose(np.linalg.det(frame[..., :3, :3]), 1.))

    return frame


def get_frame_for_rays(r1, r2):
    """Construct a local coordinate frame from two input rays and
    return it.

    Notes:
        Rays are represented using homogeneous coordinates with a
        point and unit direction. This function constructs a coordinate
        frame using the direction of r1 as the x-direction and considers
        the point of r2 to lie in the xy-plane.

    Args:
        r1 (numpy.array): A (2, 4) array representing a ray in space
            with a point and a unit direction.
        r2 (numpy.array): A (2, 4) array representing a ray in space
            with a point and a unit direction.

    Returns:
        numpy.array: A (4, 4) array representing the coordinate frame
        as a homogenous transform.
    """
    trans = r1[0]
    x_hat = r1[1]
    assert(np.isclose(np.linalg.norm(x_hat), 1.))  # make sure this is unity

    xy_pojnt = r2[0] - trans
    y_component = xy_pojnt - (np.dot(xy_pojnt, x_hat) * x_hat)
    y_hat = y_component / np.linalg.norm(y_component)

    z_hat = np.zeros(len(x_hat))
    z_hat[:-1] = np.cross(x_hat[:-1], y_hat[:-1])  # right-handed system
    assert(np.isclose(np.linalg.norm(z_hat), 1.))  # sanity check

    # construct frame
    return np.column_stack((x_hat, y_hat, z_hat, trans))


def main(argv):
    """
    """
    opts = ['-add_orbitals']
    target_fname = '../DHR10_133144_1.pdb'  # test!

    pyrosetta.init(extra_options=' '.join(opts),
                   set_logging_handler='interactive')
    p = pyrosetta.pose_from_file(target_fname)
    hbset = find_hbonds(p)
    surf_hbset = identify_bonded_pairs(p, hbset)

    # make a list of the rays and then concatenate them with np.stack()
    donors = []
    acceptors = []
    for hbond in surf_hbset.hbonds():
        don_ray, acc_ray = find_ray_pair_for_hbond(p, hbond)
        donors.append(don_ray)
        acceptors.append(acc_ray)

    hash_rays(np.stack(donors), np.stack(acceptors))


if __name__ == '__main__':
    main(sys.argv)
