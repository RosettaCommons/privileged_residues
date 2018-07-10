import h5py
import numpy as np
import pandas

class GenericTable:

    def __init__(self, dbpath):
        self._table = h5py.File(dbpath, "r")

        labels = [ ]
        def init_labels(name, item):
            nonlocal labels
            if (isinstance(item, h5py.Dataset)):
                labels.append(name)
        
        self._table.visititems(init_labels)

        self._labels = labels
    
        self._indices = { }
        
    def __getitem__(self, key):
        return self.fetch(*key) if isinstance(key, tuple) else self.fetch(key)
                
    def fetch(self, key, findgroup = ""):
        data = [ ]

        for label in self._labels:
            if (not findgroup or findgroup in label):
                dataset = self._table[label]
                if (label not in self._indices):
                    self._indices[label] = pandas.Index(dataset[dataset.dtype.names[0]])

                index = self._indices[label]

                if (key in index):
                    results = index.get_loc(key)
                    data.extend(dataset[results])

        return np.array(data)

    def __iter__(self):
        for label in self._labels:
            yield from self._table[label]

    def __len__(self):
        return sum([len(self._table[label]) for label in self._labels])

