import h5py
import numpy as np
import pandas

class GenericTable:
    """Indexed key-value store implementation.

    Best used on large datasets that do not fit into memory.
    """

    def __init__(self, dbpath):
        """
        Args:
            dbpath (str): Path to HDF5 database.
        """

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
        """
        Parameters
        ----------
        key : np.uint64 or tuple(np.uint64, str)
            Either a hash value or a tuple containing a hash and a named
            group to search in.

        Returns
        -------
        np.ndarray
            Concatenated list of matches for a hash (and group) query.
        """

        return self.fetch(*key) if isinstance(key, tuple) else self.fetch(key)
                
    def fetch(self, key, findgroup = ""):
        """
        Parameters
        ----------
        key : np.uint64
            A hash value.

        findgroup : str, optional
            A named group to search for hashes in. Defaults to "", which
            searches in all named groups.

        Returns
        -------
        np.ndarray
            Concatenated list of matches for a hash and group query.
        """

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
        """
        Yields
        ------
        np.ndarray
            A record from the database.
        """

        for label in self._labels:
            yield from self._table[label]

    def __len__(self):
        """
        Returns
        -------
        int
            The total number of records in the database.
        """

        return sum([len(self._table[label]) for label in self._labels])

