from ctree.frontend import get_ast
from .basic_blocks import get_basic_block, separate_composable_blocks, \
    process_composable_blocks
import inspect
import sys
# from copy import deepcopy
import ast


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
                name = original_ast.body[0].args.args[index]
                if sys.version_info >= (3, 0):
                    symbol_table[name.arg] = arg
                else:
                    symbol_table[name.id] = arg
        basic_block = separate_composable_blocks(orig_basic_block, symbol_table)
        basic_block = process_composable_blocks(basic_block, symbol_table)
        callable = get_callable(basic_block, symbol_table, args)
        return callable(*args, **kwargs)

    return meta_specialized


def my_exec(func, env):
    if sys.version_info >= (3, 0):
        exec(func, env)
    else:
        exec(func) in env


def get_callable(basic_block, env, args):
    if sys.version_info >= (3, 0):
        tree = ast.Module(
            [ast.FunctionDef(basic_block.name, basic_block.params,
                             list(basic_block.body), [], None)]
        )
    else:
        tree = ast.Module(
            [ast.FunctionDef(basic_block.name, basic_block.params,
                             list(basic_block.body), [])]
        )
    ast.fix_missing_locations(tree)
    my_exec(compile(tree, filename="tmp", mode="exec"), env)
    return env[basic_block.name]
