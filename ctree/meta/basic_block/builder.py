import ast
from functools import reduce
from .nodes import Assign, Symbol, Op, Return, Param, Constant


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
        string = "BasicBlock\n"
        string += "  Name: {}\n".format(self._name)
        string += "  Params: {}\n".format(", ".join(map(str, self._params)))
        string += "  Body:\n    {}".format("\n    ".join(map(str, self._body)))
        return string


class TmpGenerator(object):
    def __init__(self):
        self._curr = -1

    def __call__(self):
        self._curr += 1
        return Symbol("_t{}".format(self._curr))


class BlockDecomposer(object):
    def __init__(self):
        self.gen_tmp = TmpGenerator()

    def visit(self, expr, curr_target=None):
        if isinstance(expr, ast.Return):
            tmp = self.gen_tmp()
            body = self.visit(expr.value, tmp)
            body.append(Return(tmp))
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
                    body.append(self.visit(operand, tmp))
                    operands.append(tmp)
            if isinstance(expr.op, ast.Add):
                op = operands[0].name + '.__add__'
            elif isinstance(expr.op, ast.Mult):
                op = operands[0].name + '.__mul__'
            else:
                raise Exception("Unsupported operation")
            body.append(Assign(curr_target, Op(op, operands)))
        elif isinstance(expr, ast.Assign):
            target = Symbol(expr.targets[0].id)
            body = self.visit(expr.value, target)
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body


def get_basic_block(module):
    func = module.body[0]
    decomposer = BlockDecomposer()
    params = [Param(arg.arg) for arg in func.args.args]
    body = map(decomposer.visit, func.body)
    body = reduce(lambda x, y: x + y, body, [])
    block = BasicBlock(func.name, params, body)
    print(block)
    return block
