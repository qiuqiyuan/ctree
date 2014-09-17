import unittest
from ctree.frontend import get_ast

from ctree.meta.basic_block.builder import get_basic_block
from ctree.meta.basic_block.builder import Return


class TestBasicBlockBuilder(unittest.TestCase):
    def test_simple_return(self):
        def func(a, b):
            return a + b

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        self.assertEqual(len(basic_block), 2)
        self.assertEqual(basic_block[0].target.name, '_t0')
        self.assertEqual(
            basic_block[0].value.op,
            'a.__add__'
        )
        expected = ['a', 'b']
        for index, arg in enumerate(basic_block[0].value.args):
            self.assertEqual(arg.name, expected[index])
        self.assertIsInstance(basic_block[1], Return)
        self.assertEqual(basic_block[1].value.name, '_t0')

    def test_simple_body(self):
        def func(a, b):
            c = a * b
            return c * 3

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].target.name, 'c')
        self.assertEqual(
            basic_block[0].value.op,
            'a.__mul__'
        )
        expected = ['a', 'b']
        for index, arg in enumerate(basic_block[0].value.args):
            self.assertEqual(arg.name, expected[index])
        self.assertEqual(basic_block[1].target.name, '_t0')
        self.assertEqual(
            basic_block[1].value.op,
            'c.__mul__'
        )
        self.assertEqual(basic_block[1].value.args[0].name, 'c')
        self.assertEqual(basic_block[1].value.args[1].value, 3)
        self.assertIsInstance(basic_block[2], Return)
        self.assertEqual(basic_block[2].value.name, '_t0')
