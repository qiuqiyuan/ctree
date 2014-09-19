from ctree.frontend import get_ast
from .basic_blocks import get_basic_block, find_composable_blocks, process_composable_blocks, get_callable
import inspect
import sys
from copy import deepcopy


def meta(func):
    original_ast = get_ast(func)
    orig_basic_block = get_basic_block(original_ast)

    def meta_specialized(*args, **kwargs):
        symbol_table = dict(func.__globals__, **kwargs)
        # TODO: This should be done lazily as symbols are needed
        # coul be problematic/slow with a large stack
        for frame in inspect.stack()[1:]:
            symbol_table.update(frame[0].f_locals)
            for index, arg in enumerate(args):
                if sys.version_info >= (3, 0):
                    symbol_table[original_ast.body[0].args.args[index].arg] = arg
                else:
                    symbol_table[original_ast.body[0].args.args[index].id] = arg
        basic_block = find_composable_blocks(orig_basic_block, symbol_table)
        basic_block = process_composable_blocks(basic_block, symbol_table)
        callable = get_callable(basic_block, symbol_table, args)
        return callable(*args, **kwargs)

    return meta_specialized
