import h5py
import pandas

from typing import Iterable, MappingView

# NOTE(onalant): <C-g> will show current file in nvi!

class ResidueTable(Mapping[np.uint64, Iterable[pyrosetta.Pose]]):

    def __init__(self, cart_resl: float = 0.1, ori_resl: float = 2.0, cart_bound: float = 16.0) -> None:
        self._raygrid = RayToRay4dHash(ori_resl, _LEVER, bound=_BOUND)
        self._lattice = XformHash_bt24_BCC6_X3f(cart_resl, ori_resl, cart_bound)
        self._table = None
        
        self.__restypeset = ChemicalManager.get_instance().residue_type_set(_TOPOLOGY)
        self.__dummypose = pyrosetta.pose_from_sequence("A", _TOPOLOGY)

    def __get_item__(self, key: np.uint64) -> Iterable[pyrosetta.Pose]:
        placements = self._table[key]

        for placement in placements:
            position = self._lattice.get_center([placement[1]])['raw'].squeeze()
            fgrp_info = functional_groups[placement[0]]
            resname = fgrp_info[0]
            fgroup = fgrp_info[1]

            res = ResidueFactory.create_residue(self.__restypeset.name_map(resname))

            self.__dummypose.replace_residue(1, res, False)
            
            coords = np.stack([np.array(list(self.__dummypose.residue(1).xyz(atom))) for atom in fgroup.atoms])
            stub = geometry.coords_to_transform(coords)

            transform = np.dot(position, np.linalg.inv(stub))
            self.__dummypose.apply_transform(transform)
            yield self.__dummypose.clone()

    def __iter__(self):
        return iter(self._table)

    def __len__(self):
        return len(self.table)

