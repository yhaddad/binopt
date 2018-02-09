#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_binopt
----------------------------------

Tests for `binopt` module.
"""


import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

import binopt
import numpy  as np
from scipy import special as sp
from scipy.special import logit, expit

class test_binopt(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        sevent = 1000
        bevent = 10000
        X = np.concatenate(
            (
                expit(np.random.normal(+2.0, 2.0, sevent)),
                expit(np.random.normal(-0.5, 2.0, bevent))
            )
        )
        Y = np.concatenate((
            np.ones(sevent),
            np.zeros(bevent)
        ))
        W = np.concatenate((np.ones(sevent), np.ones(bevent)))

        binner = binopt.optimize_bin(
            nbins=3, range=[0, 1],
            drop_last_bin=True,
            fix_upper=True,
            fix_lower=False,
            use_kde_density=True
        )
        opt = binner.fit(X, Y, sample_weights=W,
                         method="Nelder-Mead",
                         breg=None, fom="AMS2")
        self.assertEqual(opt.x.shape[0], 3)

    def test_000_something(self):
        pass

if __name__ == '__main__':
    unittest.main()
