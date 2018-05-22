import pyrosetta
import table

import chemical
import geometry
import util

from rif.geom.ray_hash import RayToRay4dHash
from rif.hash import XformHash_bt24_BCC6_X3f

from pyrosetta.rosetta.core.conformation import ResidueFactory

# WARN(onalant): Constants recommended by Will Sheffler; do not change unless /absolutely/ sure
LEVER = 10
BOUND = 1000
MODE = "fa_standard"

class PrivilegedResidues:

	def __init__(self, path = "/home/onalant/dump/2018-05-07_datatables/database.h5", lru = 0):
		self._data = table.GenericTable(path, lru)

		cart_resl = self._data._table.attrs["cart_resl"]
		ori_resl = self._data._table.attrs["ori_resl"]
		cart_bound = self._data._table.attrs["cart_bound"]

		self._lattice = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
		self._raygrid = RayToRay4dHash(ori_resl, LEVER, bound=BOUND)

	# bidentate: "sc_sc", "sc_scbb", "sc_bb"
	# network: "acceptor_acceptor", "acceptor_donor", "donor_acceptor", "donor_donor"
	def match(ray1, ray2, groups=["bidentate"]):
		dummy_pose = pyrosetta.pose_from_sequence("A", "fa_standard")
		res_type_set = dummy_pose.conformation().residue_type_set_for_conf()

		def _numpy_to_rif_ray(r):
			return r.astype('f4').reshape(r.shape[:-2] + (8,)).view(rif.geom.Ray)

		hashed_rays = int(self._raygrid.get_keys(*(_numpy_to_rif_ray(r) for r in [r1, r2])).squeeze())

		for group in groups:
			results = self._data[hashed_rays, group]
			ray_frame = geometry.rays_to_transform(ray1, ray2)

			for pos_info in results:
				stored_frame = self._lattice.get_center([pos_info[2]])["raw"].squeeze()
				resname = pos_info[1].decode("utf-8")
				pos_grp = chemical.functional_groups[resname]

				dummy_pose.replace_residue(1, ResidueFactory.create_residue(res_type_set.name_map(resname), False)

				coords = [np.array([*dummy_pose.residues[1].xyz(atom)]) for atom in pos_grp.atoms]
				c = np.stack(coords)

				pos_frame = geometry.coords_to_transform(c)
				final = np.dot(np.dot(ray_frame, stored_frame), np.linalg.inv(pos_frame))
				dummy_pose.apply_transform(final)
				yield dummy_pose.clone()

