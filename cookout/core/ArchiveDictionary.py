#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class to save recent values in memory and periodically cache some older
values to disk.
"""
import collections
import copy
import os
import pickle
import shutil
import tempfile
import weakref


class ArchiveDictionary(object):
    """Dictionary which caches recent values and flush some older values
    to disk.
    """
    def __init__(self, archive_frequency=10, cache_size=5):
        """Upon garbage collection, an ArchiveDictionary will delete the
        temporary directory used to store the archived values.

        Arguments:
        - `archive_frequency`: Integer specifying how often to save a
          provided solution. Defaults to 10 (so save the zeroeth,
          tenth, etc. input).
        - `cache_size`: Integer specifying how many previous solutions
          to save; that is, at any point the last cache_size inputs are
          available. Defaults to 5.
        """
        self._archive_frequency = archive_frequency
        self._save_count = 0
        self._cache_size = cache_size
        self._prefix = "archivedictionary_"

        try:
            self._archive_directory = tempfile.mkdtemp(dir="/scratch/drwells",
                                                       prefix=self._prefix)
        # /scratch may not be available depending on the machine
        except OSError:
            self._archive_directory = tempfile.mkdtemp(prefix=self._prefix)

        self._archive_directory = self._archive_directory + os.sep
        self._archive_fnames = dict()
        self._cache = collections.OrderedDict()

        self._wr = weakref.ref(self, lambda wr, f=self._archive_directory:
                               shutil.rmtree(f))

    def cache_keys(self):
        """
        Return an iterator over the keys stored in memory.
        """
        return iter(self._cache.keys())

    def archive_keys(self):
        """
        Return an iterator over the keys stored to disk.
        """
        return iter(self._archive_fnames.keys())

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        elif key in self._archive_fnames:
            with open(self._archive_fnames[key]) as fname:
                return pickle.load(fname)
        else:
            raise KeyError

    def __setitem__(self, key, value):
        value = copy.copy(value)
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

        if self._save_count % self._archive_frequency == 0:
            path = self._archive_directory + "arch_" + str(key) + ".pkl"
            if path in self._archive_fnames:
                raise ValueError("Support for keys without unique string "
                                 "representations is poor. This is a bug.")
            self._archive_fnames[key] = path
            with open(path, 'w') as fname:
                pickle.dump(value, fname)
        self._save_count += 1
