from ctree.frontend import get_ast
from .basic_blocks import get_basic_block
import inspect
import sys
from copy import deepcopy


def meta(func):
    basic_block = get_basic_block(get_ast(func))

    def meta_specialized(*args, **kwargs):
        symbol_table = dict(func.__globals__, **kwargs)
        # TODO: This should be done lazily as symbols are needed
        # coul be problematic/slow with a large stack
        for frame in inspect.stack()[1:]:
            symbol_table.update(frame[0].f_locals)
            for index, arg in enumerate(args):
                if sys.version_info >= (3, 0):
                    symbol_table[self._original_ast.body[0].args.args[index]] = arg
                else:
                    symbol_table[self._original_ast.body[0].args.args[index].id] = arg
        basic_block = find_composable_blocks(self.basic_block, symbol_table)
        return execute_basic_block(basic_block)

    return meta_specialized
