import ast
from functools import reduce
from .nodes import Assign, Symbol, Return, Param, Constant, FunctionCall
import sys


class BasicBlock(object):
    def __init__(self, name, params, body):
        self._name = name
        self._params = params
        self._body = body

    @property
    def body(self):
        return self._body

    def __len__(self):
        return len(self._body)

    def __getitem__(self, item):
        return self._body[item]

    def __repr__(self):
        return """
BasicBlock
  Name: {name}
  Params: {params}
  Body:
    {body}
""".format(name=self._name,
           params=", ".join(map(str, self._params)),
           body="\n    ".join(map(str, self._body)))


class BlockDecomposer(object):
    def __init__(self):
        self._curr = -1

    def gen_tmp(self):
        self._curr_tmp += 1
        return Symbol("_t{}".format(self._curr))

    def visit(self, expr, curr_target=None):
        if isinstance(expr, ast.Return):
            tmp = self.gen_tmp()
            body = self.visit(expr.value, tmp)
            body.append(Return(tmp))
        elif isinstance(expr, ast.Name):
            return Symbol(expr.id)
        elif isinstance(expr, ast.BinOp):
            body = []
            operands = []
            for operand in [expr.left, expr.right]:
                if isinstance(operand, ast.Name):
                    operands.append(Symbol(operand.id))
                elif isinstance(operand, ast.Num):
                    operands.append(Constant(operand.n))
                else:
                    tmp = self.gen_tmp()
                    body.extend(self.visit(operand, tmp))
                    operands.append(tmp)
            if isinstance(expr.op, ast.Add):
                op = operands[0].name + '.__add__'
            elif isinstance(expr.op, ast.Mult):
                op = operands[0].name + '.__mul__'
            elif isinstance(expr.op, ast.Sub):
                op = operands[0].name + '.__sub__'
            elif isinstance(expr.op, ast.Div):
                op = operands[0].name + '.__div__'
            else:
                raise Exception("Unsupported BinOp {}".format(expr.op))
            body.append(Assign(curr_target,
                               FunctionCall(Symbol(op), operands)))
        elif isinstance(expr, ast.Assign):
            target = Symbol(expr.targets[0].id)
            body = self.visit(expr.value, target)
        elif isinstance(expr, ast.Call):
            body = []
            args = []
            for arg in expr.args:
                val = self.visit(arg)
                if isinstance(val, list):
                    tmp = self.gen_tmp()
                    val = self.visit(arg, tmp)
                    body.extend(val)
                    args.append(tmp)
                elif isinstance(val, (Symbol, Constant)):
                    args.append(val)
                else:
                    raise Exception("Call argument returned\
                                     unsupported type {}".format(type(val)))
            if curr_target is not None:
                body.append(Assign(curr_target,
                                   FunctionCall(self.visit(expr.func), args)))
            else:
                body.append(FunctionCall(self.visit(expr.func), args))
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body


def get_basic_block(module):
    func = module.body[0]
    decomposer = BlockDecomposer()
    if sys.version_info > (3, 0):
        params = [Param(arg.arg) for arg in func.args.args]
    else:
        params = [Param(arg.id) for arg in func.args.args]
    body = map(decomposer.visit, func.body)
    body = reduce(lambda x, y: x + y, body, [])
    return BasicBlock(func.name, params, body)
