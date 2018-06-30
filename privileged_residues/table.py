import h5py
import numpy as np
import pandas

class GenericTable:

    def __init__(self, dbpath):
        self._table = h5py.File(dbpath, "r")
    
        self._indices = { }
        
    def __getitem__(self, key):
        if (isinstance(key, int)):
            return self.fetch(key)
        elif (isinstance(key, tuple)):
            return self.fetch(*key)
        else:
            raise KeyError("Must search for hash---group pair!")

    def fetch(self, key, findgroup = ""):
        data = [ ]

        # NOTE(onalant): top-level searching, just add ``name'' keys be attributes to query for searching
        def do_visit(name, dataset):
            if (not findgroup or findgroup in name):
                if (name not in self._indices):
                    self._indices[name] = pandas.Index(dataset[dataset.dtype.names[0]])

                index = self._indices[name]
                
                if (key in index):
                    results = index.get_loc(key)
                    data.append(dataset[results])

        self._visit_datasets(do_visit)

        return np.concatenate(data) if len(data) else np.array(data)

    def __iter__(self):
        datasets = [ ]

        def do_visit(name, dataset):
            nonlocal datasets
            datasets.append(dataset)

        self._visit_datasets(do_visit)
        
        for dataset in datasets:
            yield from dataset

    def __len__(self):
        totlen = 0

        def do_visit(name, dataset):
            nonlocal totlen
            totlen += len(dataset)

        self._visit_datasets(do_visit)

        return totlen


    def _visit_datasets(self, callback):
        def do_visit(name, item):
            if (isinstance(item, h5py.Dataset)):
                key = tuple(filter(len, name.split("/")))

                callback(key, item)

        self._table.visititems(do_visit)

