import unittest
from ctree.frontend import get_ast

from ctree.meta.basic_block.builder import get_basic_block
from ctree.meta.basic_block.builder import Return
from ctree.meta.basic_block.nodes import Symbol, Constant


class TestBasicBlockBuilder(unittest.TestCase):
    def _check_args(self, actual, expected):
        for act, exp in zip(actual, expected):
            if isinstance(act, Symbol):
                self.assertEqual(act.name, exp)
            elif isinstance(act, Constant) and act.value is not exp:
                self.assertEqual(act.value, exp)

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
        self._check_args(basic_block[0].value.args, ['a', 'b'])
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
        self._check_args(basic_block[0].value.args, ['a', 'b'])
        self.assertEqual(basic_block[1].target.name, '_t0')
        self.assertEqual(
            basic_block[1].value.op,
            'c.__mul__'
        )
        self._check_args(basic_block[1].value.args, ['c', 3])
        self.assertIsInstance(basic_block[2], Return)
        self.assertEqual(basic_block[2].value.name, '_t0')

    def test_unpack_expression(self):
        def func(a, b):
            return a * b + c

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].target.name, '_t1')
        self.assertEqual(
            basic_block[0].value.op,
            'a.__mul__'
        )
        self._check_args(basic_block[0].value.args, ['a', 'b'])
        self.assertEqual(basic_block[1].target.name, '_t0')
        self.assertEqual(
            basic_block[1].value.op,
            '_t1.__add__'
        )
        self._check_args(basic_block[1].value.args, ['_t1', 'c'])
        self.assertIsInstance(basic_block[2], Return)
        self.assertEqual(basic_block[2].value.name, '_t0')

    def test_unpack_precedence(self):
        def func(a, b):
            return a + b * c

        tree = get_ast(func)
        basic_block = get_basic_block(tree)
        print(basic_block)
        self.assertEqual(len(basic_block), 3)
        self.assertEqual(basic_block[0].target.name, '_t1')
        self.assertEqual(
            basic_block[0].value.op,
            'b.__mul__'
        )
        self._check_args(basic_block[0].value.args, ['b', 'c'])
        self.assertEqual(basic_block[1].target.name, '_t0')
        self.assertEqual(
            basic_block[1].value.op,
            'a.__add__'
        )
        self._check_args(basic_block[1].value.args, ['a', '_t1'])
        self.assertIsInstance(basic_block[2], Return)
        self.assertEqual(basic_block[2].value.name, '_t0')
