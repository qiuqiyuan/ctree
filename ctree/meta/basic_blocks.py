import ast
from functools import reduce
import sys
from ctree.jit import LazySpecializedFunction


def eval_in_env(env, expr):
    if isinstance(expr, ast.Name):
        return env[expr.id]
    elif isinstance(expr, ast.Attribute):
        return getattr(env[expr.value.id], expr.attr)
    raise Exception("Unhandled type for eval_in_env {}".format(type(expr)))


def str_dump(item, tab=0):
    if isinstance(item, ast.Assign):
        return "{} = {}".format(str_dump(item.targets[0]),
                                str_dump(item.value))
    elif isinstance(item, ast.Return):
        return "return {}".format(str_dump(item.value))
    elif isinstance(item, ast.Attribute):
        return "{}.{}".format(str_dump(item.value), item.attr)
    elif isinstance(item, ast.Call):
        return "{}({})".format(str_dump(item.func),
                               ", ".join(map(str_dump, item.args)))
    elif isinstance(item, ast.Num):
        return str(item.n)
    elif isinstance(item, ast.Name):
        return item.id
    elif isinstance(item, ComposableBlock):
        tab = "\n" + "".join([" " for _ in range(tab + 2)])
        return "ComposableBlock:{}{}".format(tab, tab.join(
            map(str_dump, item.statements)))
    raise Exception("Unsupport type for dumping {}: {}".format(type(item),
                                                               item))


class BasicBlock(object):
    def __init__(self, name, params, body, composable_blocks=None):
        self.name = name
        self.params = params
        self.body = body
        if composable_blocks is None:
            self.composable_blocks = ()
        else:
            self.composable_blocks = composable_blocks

    def __len__(self):
        return len(self.body)

    def __getitem__(self, item):
        return self.body[item]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """
BasicBlock
  Name: {name}
  Params: {params}
  Body:
    {body}
""".format(name=self.name,
           params=", ".join(self.params),
           body="\n    ".join(map(lambda x: str_dump(x, 4), self.body)))


def find_composable_blocks(basic_block, env):
    statements = ()
    composable_statements = ()
    composable_blocks = ()
    for statement in basic_block.body:
        if isinstance(statement, ast.Assign) and \
           isinstance(statement.value, ast.Call) and \
           isinstance(eval_in_env(env, statement.value.func),
                      LazySpecializedFunction):
            composable_statements += (statement, )
        else:
            if len(composable_statements) > 1:
                composable_block = ComposableBlock(composable_statements)
                statements += (composable_block, )
                composable_blocks += (composable_block, )
            elif len(composable_statements) == 1:
                statements += (composable_statements[0], )
            statements += (statement, )
            composable_statements = ()

    return BasicBlock(basic_block.name, basic_block.params, statements, composable_blocks)


class ComposableBlock(object):
    """docstring for ComposableBlock"""
    def __init__(self, statements):
        super(ComposableBlock, self).__init__()
        self.statements = statements


class BlockDecomposer(object):
    def __init__(self):
        self.__curr_tmp = -1

    def gen_tmp(self):
        self.__curr_tmp += 1
        return "_t{}".format(self.__curr_tmp)

    def visit(self, expr, curr_target=None):
        if isinstance(expr, ast.Return):
            tmp = self.gen_tmp()
            body = self.visit(expr.value, ast.Name(tmp, ast.Store()))
            body += (ast.Return(ast.Name(tmp, ast.Load())), )
        elif isinstance(expr, ast.Name):
            return expr
        elif isinstance(expr, ast.BinOp):
            body = ()
            operands = ()
            for operand in [expr.left, expr.right]:
                if isinstance(operand, (ast.Name, ast.Num)):
                    operands += (operand, )
                else:
                    tmp = self.gen_tmp()
                    body += self.visit(operand,
                                       ast.Name(tmp, ast.Store()))
                    operands += (ast.Name(tmp, ast.Load()), )
            if isinstance(expr.op, ast.Add):
                op = ast.Attribute(operands[0], '__add__', ast.Load())
            elif isinstance(expr.op, ast.Mult):
                op = ast.Attribute(operands[0], '__mul__', ast.Load())
            elif isinstance(expr.op, ast.Sub):
                op = ast.Attribute(operands[0], '__sub__', ast.Load())
            elif isinstance(expr.op, ast.Div):
                op = ast.Attribute(operands[0], '__div__', ast.Load())
            else:
                raise Exception("Unsupported BinOp {}".format(expr.op))
            body += (ast.Assign([curr_target],
                                   ast.Call(op, operands, [], None, None)), )
        elif isinstance(expr, ast.Assign):
            target = expr.targets[0]
            body = self.visit(expr.value, target)
        elif isinstance(expr, ast.Call):
            body = ()
            args = ()
            for arg in expr.args:
                val = self.visit(arg)
                if isinstance(val, tuple):
                    tmp = self.gen_tmp()
                    val = self.visit(arg, ast.Name(tmp, ast.Store))
                    body += val
                    args += (ast.Name(tmp, ast.Load()), )
                elif isinstance(val, (ast.Name, ast.Num)):
                    args += (val, )
                else:
                    raise Exception("Call argument returned\
                                     unsupported type {}".format(type(val)))
            if curr_target is not None:
                body += (ast.Assign(
                    [curr_target],
                    ast.Call(self.visit(expr.func), args, [], None, None)
                ), )
            else:
                body += (ast.Call(self.visit(expr.func), args), )
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body


def get_basic_block(module):
    func = module.body[0]
    decomposer = BlockDecomposer()
    if sys.version_info > (3, 0):
        params = tuple(arg.arg for arg in func.args.args)
    else:
        params = tuple(arg.id for arg in func.args.args)
    body = map(decomposer.visit, func.body)
    body = reduce(lambda x, y: x + y, body, ())
    return BasicBlock(func.name, params, body)
