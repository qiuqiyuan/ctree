import unittest

from ctree.meta.core import meta


class TestMetaDecorator(unittest.TestCase):
    def test_simple(self):
        @meta
        def func(a):
            return a + 3

        self.assertEqual(func(3), 6)
