import logging
import numpy as np
import pyrosetta

from . import chemical
from . import geometry
from . import table
from . import util

from rif.geom import Ray
from rif.geom.ray_hash import RayToRay4dHash
from rif.hash import XformHash_bt24_BCC6_X3f

from pyrosetta.rosetta.core.conformation import ResidueFactory

logging.basicConfig(level=logging.WARN)
_logging_handler = "interactive"

# WARN(onalant): Constants recommended by Will Sheffler; do not change unless /absolutely/ sure
LEVER = 10
BOUND = 1000
MODE = "fa_standard"

def _init_pyrosetta():
    from os import path

    _dir = path.join(path.dirname(__file__), "data", "functional_groups")

    param_files = [path.join(_dir, x.resName + ".params") for x in chemical.functional_groups.values()]
    opts = [
        "-corrections:beta_nov16",
        "-ignore_waters false",
        "-mute core",
        "-extra_res_fa %s" % (" ".join(param_files)),
        # "-constant_seed",
        "-output_virtual"
    ]

    pyrosetta.init(extra_options=" ".join(opts), set_logging_handler=_logging_handler)

class PrivilegedResidues:

    def __init__(self, path = "/home/onalant/dump/2018-05-07_datatables/database.h5"):
        self._data = table.GenericTable(path)

        cart_resl = self._data._table.attrs["cart_resl"]
        ori_resl = self._data._table.attrs["ori_resl"]
        cart_bound = self._data._table.attrs["cart_bound"]

        self._lattice = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
        self._raygrid = RayToRay4dHash(ori_resl, LEVER, bound=BOUND)

    # bidentate: "sc_sc", "sc_scbb", "sc_bb"
    # network: "acceptor_acceptor", "acceptor_donor", "donor_acceptor", "donor_donor"
    def match(self, ray1, ray2, group):
        dummy_pose = pyrosetta.pose_from_sequence("A", "fa_standard")
        res_type_set = dummy_pose.conformation().residue_type_set_for_conf()

        hashed_rays = np.asscalar(self._raygrid.get_keys(*(util.numpy_to_rif(r) for r in [ray1, ray2])).squeeze())

        results = self._data[hashed_rays, group]
        try:
            ray_frame = geometry.rays_to_transform(ray1, ray2)
        except:
            return []

        for pos_info in results:
            try:
                stored_frame = self._lattice.get_center([pos_info["transform"]])["raw"].squeeze()
            except:
                continue

            resname = pos_info["residue"].decode("utf-8")
            pos_grp = chemical.functional_groups[resname]

            dummy_pose.replace_residue(1, ResidueFactory.create_residue(res_type_set.name_map(resname)), False)

            coords = [np.array([*dummy_pose.residues[1].xyz(atom)]) for atom in pos_grp.atoms]
            c = np.stack(coords)

            pos_frame = geometry.coords_to_transform(c)
            final = np.dot(np.dot(ray_frame, stored_frame), np.linalg.inv(pos_frame))
            dummy_pose.apply_transform(final)
            yield dummy_pose.clone()

    # NOTE(onalant): bring your own residue selector
    def search(self, pose, groups, selector):
        pairs_of_rays = []

        if np.any([x in groups for x in ["sc_sc", "bidentate"]]):
            pairs_of_rays += chemical.sc_sc_rays(pose, selector)
        if np.any([x in groups for x in ["sc_scbb", "bidentate"]]):
            pairs_of_rays += chemical.sc_scbb_rays(pose, selector)
        if np.any([x in groups for x in ["sc_bb", "bidentate"]]):
            pairs_of_rays += chemical.sc_bb_rays(pose, selector)

        if np.any([x in groups for x in ["acceptor_acceptor", "network"]]):
            pairs_of_rays += chemical.acceptor_acceptor_rays(pose, selector)
        if np.any([x in groups for x in ["acceptor_donor", "network"]]):
            pairs_of_rays += chemical.donor_acceptor_rays(pose, selector)
        if np.any([x in groups for x in ["donor_acceptor", "network"]]):
            pairs_of_rays += chemical.donor_acceptor_rays(pose, selector)
        if np.any([x in groups for x in ["donor_donor", "network"]]):
            pairs_of_rays += chemical.donor_donor_rays(pose, selector)

        for (r1, r2) in pairs_of_rays:
            yield from self.match(r1, r2, groups)

_init_pyrosetta()

