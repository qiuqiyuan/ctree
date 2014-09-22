import unittest
import numpy as np
from ctree.meta.core import meta
from array_add import array_add


class TestMetaDecorator(unittest.TestCase):
    def test_simple(self):
        @meta
        def func(a):
            return a + 3

        self.assertEqual(func(3), 6)

    def test_dataflow(self):
        @meta
        def func(a, b):
            c = array_add(a, b)
            return array_add(c, a)

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        try:
            np.testing.assert_array_almost_equal(func(a, b), a + b + a)
        except AssertionError as e:
            self.fail("Arrays not almost equal\n{}".format(e))
