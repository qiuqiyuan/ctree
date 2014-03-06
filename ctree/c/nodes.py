"""
AST nodes for C constructs.
"""

import os
import subprocess

import logging

log = logging.getLogger(__name__)

from ctree.nodes import CtreeNode, File
from ctree.util import singleton

class CNode(CtreeNode):
    """Base class for all C nodes in ctree."""

    def codegen(self, indent=0):
        from ctree.c.codegen import CCodeGen

        return CCodeGen(indent).visit(self)

    def _to_dot(self):
        from ctree.c.dotgen import CDotGen

        return CDotGen().visit(self)


class CFile(CNode, File):
    """Represents a .c file."""

    def __init__(self, name="generated", body=None):
        if not body:
            body = []
        super(CFile, self).__init__(name, body)
        self._ext = "c"

    def get_bc_filename(self):
        return "%s.bc" % self.name

    def _compile(self, program_text, compilation_dir):
        c_src_file = os.path.join(compilation_dir, self.get_filename())
        ll_bc_file = os.path.join(compilation_dir, self.get_bc_filename())
        log.info("File for generated C: %s" % c_src_file)
        log.info("File for generated LLVM: %s" % ll_bc_file)
        log.info("Generated C program: <<<\n%s\n>>>" % program_text)

        # write program text to C file
        with open(c_src_file, 'w') as c_file:
            c_file.write(program_text)

        # call clang to generate LLVM bitcode file
        import ctree

        CC = ctree.config.get('jit', 'CC')
        CFLAGS = ctree.config.get('jit', 'CFLAGS')
        compile_cmd = "%s -emit-llvm %s -o %s -c %s" % (CC, CFLAGS, ll_bc_file, c_src_file)
        log.info("Compilation command: %s" % compile_cmd)
        subprocess.check_call(compile_cmd, shell=True)

        # load llvm bitcode
        import llvm.core

        with open(ll_bc_file, 'rb') as bc:
            ll_module = llvm.core.Module.from_bitcode(bc)
        log.info("Generated LLVM Program: <<<\n%s\n>>>" % ll_module)

        return ll_module


class Statement(CNode):
    """Section B.2.3 6.6."""
    pass


class Expression(CNode):
    """Cite me."""

    def get_type(self):
        from ctree.c.types import CTypeFetcher

        return CTypeFetcher().visit(self)


class Return(Statement):
    """Section B.2.3 6.6.6 line 4."""
    _fields = ['value']

    def __init__(self, value=None):
        self.value = value
        super(Return, self).__init__()


class If(Statement):
    """Cite me."""
    _fields = ['cond', 'then', 'elze']

    def __init__(self, cond=None, then=None, elze=None):
        self.cond = cond
        self.then = then
        self.elze = elze
        super(If, self).__init__()


class While(Statement):
    """Cite me."""
    _fields = ['cond', 'body']

    def __init__(self, cond=None, body=None):
        self.cond = cond
        self.body = body if body else []
        super(While, self).__init__()


class DoWhile(Statement):
    _fields = ['body', 'cond']

    def __init__(self, body=None, cond=None):
        self.body = body if body else []
        self.cond = cond
        super(DoWhile, self).__init__()


class For(Statement):
    _fields = ['init', 'test', 'incr', 'body']

    def __init__(self, init=None, test=None, incr=None, body=None):
        self.init = init
        self.test = test
        self.incr = incr
        self.body = body
        super(For, self).__init__()


# TODO: Decide on naming scheme for define statements
class Define(Statement):
    _fields = ['defname', 'calc']
    def __init__(self, defname=None, calc=None):
        self.defname = defname
        self.calc = calc
        super().__init__()
      
    def _requires_semicolon(self):
        return False


class FunctionCall(Expression):
    """Cite me."""
    _fields = ['func', 'args']

    def __init__(self, func=None, args=None):
        self.func = func
        self.args = args if args else []
        super(FunctionCall, self).__init__()


class Literal(Expression):
    """Cite me."""
    pass


class Constant(Literal):
    """Section B.1.4 6.1.3."""

    def __init__(self, value=None):
        self.value = value
        super(Constant, self).__init__()


class Block(Statement):
    """Cite me."""
    _fields = ['body']

    def __init__(self, body=None):
        self.body = body if body else []
        super(Block, self).__init__()


class String(Literal):
    """Cite me."""

    def __init__(self, *values):
        self.values = values
        super(String, self).__init__()


class SymbolRef(Literal):
    """Cite me."""
    _next_id = 0

    def __init__(self, name=None, sym_type=None, _global=False):
        """
        Create a new symbol with the given name. If a declaration
        type is specified, the symbol is considered a declaration
        and unparsed with the type.
        """
        self.name = name
        self.type = sym_type
        self._global = False
        super(SymbolRef, self).__init__()

    def set_global(self, value=True):
        self._global = value
        return self

    @classmethod
    def unique(cls, name="name", sym_type=None):
        """
        Factory for making unique symbols.
        """
        sym = SymbolRef("%s_%d" % (name, cls._next_id), sym_type)
        cls._next_id += 1
        return sym

    def copy(self, declare=False):
        if declare:
            return SymbolRef(self.name, self.type, self._global)
        else:
            return SymbolRef(self.name)

    def get_ctype(self):
        return self.type.as_ctype()


class FunctionDecl(Statement):
    """Cite me."""
    _fields = ['params', 'defn']

    def __init__(self, return_type=None, name=None, params=None, defn=None):
        self.return_type = return_type
        self.name = name
        self.params = params if params else []
        self.defn = defn if defn else []
        self.inline = False
        self.static = False
        self.kernel = False
        super(FunctionDecl, self).__init__()

    def get_type(self):
        from ctree.c.types import FuncType
        arg_types = [p.get_type() for p in self.params]
        return FuncType(self.return_type, arg_types)

    def set_inline(self, value=True):
        self.inline = value
        return self

    def set_static(self, value=True):
        self.static = value
        return self

    def set_kernel(self, value=True):
        self.kernel = value
        return self

    def set_typesig(self, typesig):
        assert len(typesig) == len(self.params)+1
        self.return_type = typesig[0]
        for sym, ty in zip(self.params, typesig[1:]):
            sym.type = ty
        return self


class UnaryOp(Expression):
    """Cite me."""
    _fields = ['arg']

    def __init__(self, op=None, arg=None):
        self.op = op
        self.arg = arg
        super(UnaryOp, self).__init__()


class BinaryOp(Expression):
    """Cite me."""
    _fields = ['left', 'right']

    def __init__(self, left=None, op=None, right=None):
        self.left = left
        self.op = op
        self.right = right
        super(BinaryOp, self).__init__()


class AugAssign(Expression):
    """Cite me."""
    _fields = ['target', 'value']

    def __init__(self, target=None, op=None, value=None):
        self.target = target
        self.op = op
        self.value = value
        super(AugAssign, self).__init__()


class TernaryOp(Expression):
    """Cite me."""
    _fields = ['cond', 'then', 'elze']

    def __init__(self, cond=None, then=None, elze=None):
        self.cond = cond
        self.then = then
        self.elze = elze
        super(TernaryOp, self).__init__()


class Cast(Expression):
    """doc"""
    _fields = ['value']

    def __init__(self, sym_type=None, value=None):
        self.type = sym_type
        self.value = value
        super(Cast, self).__init__()


@singleton
class Op:
    class _Op(object):
        def __str__(self):
            return self._c_str

    class PreInc(_Op):
        _c_str = "++"

    class PreDec(_Op):
        _c_str = "--"

    class PostInc(_Op):
        _c_str = "++"

    class PostDec(_Op):
        _c_str = "--"

    class Ref(_Op):
        _c_str = "&"

    class Deref(_Op):
        _c_str = "*"

    class SizeOf(_Op):
        _c_str = "sizeof"

    class Add(_Op):
        _c_str = "+"

    class AddUnary(_Op):
        _c_str = "+"

    class Sub(_Op):
        _c_str = "-"

    class SubUnary(_Op):
        _c_str = "-"

    class Mul(_Op):
        _c_str = "*"

    class Div(_Op):
        _c_str = "/"

    class Mod(_Op):
        _c_str = "%"

    class Gt(_Op):
        _c_str = ">"

    class Lt(_Op):
        _c_str = "<"

    class GtE(_Op):
        _c_str = ">="

    class LtE(_Op):
        _c_str = "<="

    class Eq(_Op):
        _c_str = "=="

    class NotEq(_Op):
        _c_str = "!="

    class BitAnd(_Op):
        _c_str = "&"

    class BitOr(_Op):
        _c_str = "|"

    class BitNot(_Op):
        _c_str = "~"

    class BitShL(_Op):
        _c_str = "<<"

    class BitShR(_Op):
        _c_str = ">>"

    class BitXor(_Op):
        _c_str = "^"

    class And(_Op):
        _c_str = "&&"

    class Or(_Op):
        _c_str = "||"

    class Not(_Op):
        _c_str = "!"

    class Comma(_Op):
        _c_str = ","

    class Dot(_Op):
        _c_str = "."

    class Arrow(_Op):
        _c_str = "->"

    class Assign(_Op):
        _c_str = "="

    class ArrayRef(_Op):
        _c_str = "??"


# ---------------------------------------------------------------------------
# factory routines for building UnaryOps, BinaryOps, etc.

def PreInc(a):
    return UnaryOp(Op.PreInc(), a)


def PreDec(a):
    return UnaryOp(Op.PreDec(), a)


def PostInc(a):
    return UnaryOp(Op.PostInc(), a)


def PostDec(a):
    return UnaryOp(Op.PostDec(), a)


def BitNot(a):
    return UnaryOp(Op.BitNot(), a)


def Not(a):
    return UnaryOp(Op.Not(), a)


def Ref(a):
    return UnaryOp(Op.Ref(), a)


def Deref(a):
    return UnaryOp(Op.Deref(), a)


def SizeOf(a):
    return UnaryOp(Op.SizeOf(), a)


def Add(a, b=None):
    if b is not None:
        return BinaryOp(a, Op.Add(), b)
    else:
        return UnaryOp(Op.AddUnary(), a)


def Sub(a, b=None):
    if b is not None:
        return BinaryOp(a, Op.Sub(), b)
    else:
        return UnaryOp(Op.SubUnary(), a)


def Mul(a, b):
    return BinaryOp(a, Op.Mul(), b)


def Div(a, b):
    return BinaryOp(a, Op.Div(), b)


def Mod(a, b):
    return BinaryOp(a, Op.Mod(), b)


def Gt(a, b):
    return BinaryOp(a, Op.Gt(), b)


def Lt(a, b):
    return BinaryOp(a, Op.Lt(), b)


def GtE(a, b):
    return BinaryOp(a, Op.GtE(), b)


def LtE(a, b):
    return BinaryOp(a, Op.LtE(), b)


def Eq(a, b):
    return BinaryOp(a, Op.Eq(), b)


def NotEq(a, b):
    return BinaryOp(a, Op.NotEq(), b)


def BitAnd(a, b):
    return BinaryOp(a, Op.BitAnd(), b)


def BitOr(a, b):
    return BinaryOp(a, Op.BitOr(), b)


def BitShL(a, b):
    return BinaryOp(a, Op.BitShL(), b)


def BitShR(a, b):
    return BinaryOp(a, Op.BitShR(), b)


def BitXor(a, b):
    return BinaryOp(a, Op.BitXor(), b)


def And(a, b):
    return BinaryOp(a, Op.And(), b)


def Or(a, b):
    return BinaryOp(a, Op.Or(), b)


def Comma(a, b):
    return BinaryOp(a, Op.Comma(), b)


def Dot(a, b):
    return BinaryOp(a, Op.Dot(), b)


def Arrow(a, b):
    return BinaryOp(a, Op.Arrow(), b)


def Assign(a, b):
    return BinaryOp(a, Op.Assign(), b)


def ArrayRef(a, b):
    return BinaryOp(a, Op.ArrayRef(), b)


def AddAssign(a, b):
    return AugAssign(a, Op.Add(), b)


def SubAssign(a, b):
    return AugAssign(a, Op.Sub(), b)


def MulAssign(a, b):
    return AugAssign(a, Op.Mul(), b)


def DivAssign(a, b):
    return AugAssign(a, Op.Div(), b)


def ModAssign(a, b):
    return AugAssign(a, Op.Mod(), b)


def BitXorAssign(a, b):
    return AugAssign(a, Op.BitXor(), b)


def BitAndAssign(a, b):
    return AugAssign(a, Op.BitAnd(), b)


def BitOrAssign(a, b):
    return AugAssign(a, Op.BitOr(), b)


def BitShLAssign(a, b):
    return AugAssign(a, Op.BitShL(), b)


def BitShRAssign(a, b):
    return AugAssign(a, Op.BitShR(), b)
