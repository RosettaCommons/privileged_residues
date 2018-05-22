import h5py
import numpy as np
import pandas

from lru import LRU
from typing import Callable, Iterable, Mapping, Tuple, Union

# NOTE(onalant): <C-g> will show current file in nvi!

class GenericTable(Mapping[np.uint64, np.ndarray]):

    def __init__(self, dbpath: str, lru: int = 0) -> None:
        assert(lru >= 0)
        self._table = h5py.File(dbpath, "r")
    
        if (lru == 0):
            self._indices = { }
        else:
            self._indices = LRU(lru)
        
    def __getitem__(self, key: Union[np.uint64, Tuple[np.uint64, str]]) -> np.ndarray:
        if (isinstance(key, int)):
            return self.fetch(key)
        elif (isinstance(key, tuple)):
            return self.fetch(*key)
        else:
            raise KeyError("Must search for hash---group pair!")

    def fetch(self, key: np.uint64, findgroup: str = "") -> np.ndarray:
        data = [ ]

        # NOTE(onalant): top-level searching, just add ``name'' keys be attributes to query for searching
        def do_visit(name: Tuple[str, ...], dataset: h5py.Dataset) -> None:
            if (not findgroup or findgroup in name):
                if (name not in self._indices):
                    self._indices[name] = pandas.Index(dataset[dataset.dtype.names[0]])

                index = self._indices[name]
                
                if (key in index):
                    results = index.get_loc(key)
                    data.append(dataset[results])

        self._visit_datasets(do_visit)

        return np.concatenate(data) if len(data) else np.array(data)

    def __iter__(self) -> Iterable[np.ndarray]:
        datasets = [ ]

        def do_visit(name: Tuple[str, ...], dataset: h5py.Dataset):
            nonlocal datasets
            datasets.append(dataset)

        self._visit_datasets(do_visit)
        
        for dataset in datasets:
            yield from dataset

    def __len__(self) -> int:
        totlen = 0

        def do_visit(name: Tuple[str, ...], dataset: h5py.Dataset):
            nonlocal totlen
            totlen += len(dataset)

        self._visit_datasets(do_visit)

        return totlen


    def _visit_datasets(self, callback: Callable[[Tuple[str, ...], h5py.Dataset], None]) -> None:
        def do_visit(name: str, item: h5py.HLObject) -> None:
            if (isinstance(item, h5py.Dataset)):
                key = tuple(filter(len, name.split("/")))

                callback(key, item)

        self._table.visititems(do_visit)

