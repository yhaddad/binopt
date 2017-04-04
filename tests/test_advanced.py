# -*- coding: utf-8 -*-

from .context import zbinner

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(zbinner.binopt())


if __name__ == '__main__':
    unittest.main()
