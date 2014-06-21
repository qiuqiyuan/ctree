"""
Parses the python AST below, transforms it to C, JITs it, and runs it.
"""
import ast
import logging
import time

import numpy as np

from ctree.analyses import VerifyOnlyCtreeNodes
from ctree.c.nodes import (
    FunctionDecl, SymbolRef, For, Lt, Assign, Constant, PostInc, ArrayRef,
    FunctionCall, CFile, Op
)
from ctree.c.types import NdPointer, Void, FuncType, Int
from ctree.nodes import Project
from ctree.frontend import get_ast
from ctree.jit import LazySpecializedFunction
from ctree.templates.nodes import StringTemplate
from ctree.transformations import PyBasicConversions
from ctree.types import get_ctree_type


logging.basicConfig(level=20)


# ---------------------------------------------------------------------------
# Specializer code


class OpTranslator(LazySpecializedFunction):

    def args_to_subconfig(self, args):
        """
        Analyze arguments and return a 'subconfig', a hashable object
        that classifies them. Arguments with identical subconfigs
        might be processed by the same generated code.
        """
        A = args[0]
        return {
            'A_len': len(A),
            'A_dtype': A.dtype,
            'A_ndim': A.ndim,
            'A_shape': A.shape,
        }

    def get_declarations(self, args):

        arg_config = self.args_to_subconfig(args)

        tree = PyBasicConversions().visit(self.original_tree.body[0])

        A_dtype = arg_config['A_dtype']

        inner_type = get_ctree_type(A_dtype)
        apply_one_typesig = FuncType(inner_type, [inner_type, inner_type])

        apply_one = tree.find(FunctionDecl, name="apply")
        apply_one.set_static().set_inline()
        apply_one.set_typesig(apply_one_typesig)
        return tree

    def transform(self, py_ast, program_config):
        """
        Convert the Python AST to a C AST according to the directions
        given in program_config.
        """
        arg_config, tuner_config = program_config
        len_A = arg_config['A_len']
        A_dtype = arg_config['A_dtype']
        A_ndim = arg_config['A_ndim']
        A_shape = arg_config['A_shape']

        inner_type = get_ctree_type(A_dtype)
        array_type = NdPointer(A_dtype, A_ndim, A_shape)
        apply_one_typesig = FuncType(inner_type, [inner_type, inner_type])

        tree = CFile(
            "generated", [
                py_ast.body[0], FunctionDecl(
                    Void(), "apply_elementwise", params=[
                        SymbolRef("A", array_type),
                        SymbolRef("B", array_type),
                        SymbolRef("C", array_type)
                    ], defn=[
                        For(
                            Assign(
                                SymbolRef("i", Int()),
                                Constant(0)
                            ),
                            Lt(SymbolRef("i"), Constant(len_A)),
                            PostInc(SymbolRef("i")),
                            [Assign(
                                ArrayRef(
                                    SymbolRef("C"), SymbolRef("i")
                                ),
                                FunctionCall(
                                    SymbolRef("apply"),
                                    [ArrayRef(SymbolRef("A"), SymbolRef("i")),
                                     ArrayRef(SymbolRef("B"), SymbolRef("i"))]
                                )
                            )]
                        )]
                )
            ]
        )

        tree = PyBasicConversions().visit(tree)

        apply_one = tree.find(FunctionDecl, name="apply")
        apply_one.set_static().set_inline()
        apply_one.set_typesig(apply_one_typesig)

        entry_point_typesig = tree.find(
            FunctionDecl,
            name="apply_elementwise").get_type().as_ctype()

        return Project([tree]), entry_point_typesig

    def get_semantic_model(self, args):
        arg_config = self.args_to_subconfig(args)

        len_A = arg_config['A_len']
        A_dtype = arg_config['A_dtype']
        A_ndim = arg_config['A_ndim']
        A_shape = arg_config['A_shape']

        array_type = NdPointer(A_dtype, A_ndim, A_shape)

        tree = FunctionDecl(
            Void(), "apply_elementwise", params=[
                SymbolRef("A", array_type),
                SymbolRef("B", array_type),
                SymbolRef("C", array_type)
            ], defn=[
                For(
                    Assign(
                        SymbolRef("i", Int()),
                        Constant(0)
                    ),
                    Lt(SymbolRef("i"), Constant(len_A)),
                    PostInc(SymbolRef("i")),
                    [Assign(
                        ArrayRef(
                            SymbolRef("C"), SymbolRef("i")
                        ),
                        FunctionCall(
                            SymbolRef("apply"),
                            [ArrayRef(SymbolRef("A"), SymbolRef("i")),
                             ArrayRef(SymbolRef("B"), SymbolRef("i"))]
                        )
                    )]
                )]
        )

        entry_point_typesig = tree.find(
            FunctionDecl,
            name="apply_elementwise"
        ).get_type().as_ctype()

        return tree, entry_point_typesig


class ElementwiseArrayOp(object):

    def __init__(self):
        """Instantiate translator."""
        self.c_apply_elementwise = OpTranslator(
            get_ast(self.apply), "apply_elementwise"
        )

    def __call__(self, A, B, C):
        """Apply the operator to the arguments via a generated function."""
        return self.c_apply_elementwise(A, B, C)


class Fuser(object):

    def __init__(self):
        self.body = self.fuse
        self.fuse = self.shadow_fuse
        self.blocks = []
        self.type_sigs = []
        self.args = ()
        self.declarations = []
        self.arg_names = []
        self.fn = None

    def shadow_fuse(self):
        if self.fn is not None:
            # Primitive caching
            return self.fn(*self.args)
        tree = get_ast(self.body)
        BlockBuilder(self).visit(tree)
        visitor = UniqueNamer(set())
        for index in range(len(self.blocks)):
            visitor.visit(self.blocks[index])
            visitor.visit(self.declarations[index])
            visitor = UniqueNamer(visitor.seen.union(visitor.prev_seen))
        main_file = CFile('generated', [])

        # Fuse file bodies
        for block in self.blocks:
            main_file.body.append(block)

        # Fuse functions into one entry point
        FuseFunctions().visit(main_file)

        # Fuse any fusable loops
        FuseLoops().visit(main_file)

        # Find arguments that can be promoted to registers
        self.args = list(self.args)
        replace_args = []
        for index1, arg1 in enumerate(self.arg_names):
            for index2, arg2 in enumerate(self.arg_names[index1 + 1:]):
                if arg1 == arg2:
                    replace_args.append((index1, 1 + index1 + index2))

        tmp = []
        # Promote arguments to registers
        for replace in replace_args:
            ArgReplacer(replace).visit(main_file.body[0])
            tmp.extend(list(replace))

        # Remove promoted arguments
        for index in sorted(tmp, reverse=True):
            del main_file.body[0].params[index]
            del self.args[index]

        for declaration in self.declarations:
            main_file.body.insert(0, declaration)
        # Compile and call fused function
        # TODO: This should be handle by composer interface
        proj = Project([main_file])
        VerifyOnlyCtreeNodes().visit(proj)
        typesig = proj.find(
            FunctionDecl,
            name='apply_elementwise').get_type().as_ctype()
        self.module = proj.codegen()
        self.fn = self.module.get_callable('apply_elementwise', typesig)
        self.fn(*self.args)

    def append_block(self, func, *args):
        self.args += args
        tree, type_sig = func.c_apply_elementwise.get_semantic_model(args)
        self.declarations.append(
            func.c_apply_elementwise.get_declarations(args))
        self.blocks.append(tree)
        self.type_sigs.append(type_sig)


class ArgReplacer(ast.NodeTransformer):
    unique_num = 0

    def __init__(self, to_replace):
        self.indices = to_replace
        self.declared = False
        self.to_replace = {}
        self.tmp_name = "tmp{0}".format(ArgReplacer.unique_num)
        ArgReplacer.unique_num += 1

    def visit_FunctionDecl(self, node):
        for i, index in enumerate(self.indices):
            param = node.params[index]
            self.to_replace[param.name] = param.type.get_base_type()

        node.defn = map(self.visit, node.defn)
        return node

    def visit_BinaryOp(self, node):
        if isinstance(node.op, Op.ArrayRef):
            if node.left.name in self.to_replace:
                if not self.declared:
                    self.declared = True
                    return SymbolRef(
                        self.tmp_name,
                        self.to_replace[
                            node.left.name])
                else:
                    return SymbolRef(self.tmp_name)
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node


class UniqueNamer(ast.NodeTransformer):
    unique_num = 0

    def __init__(self, prev_seen):
        super(UniqueNamer, self).__init__()
        UniqueNamer.unique_num += 1
        self.prev_seen = prev_seen
        self.seen = set()
        self.new_name_map = {}

    def visit_FunctionDecl(self, node):
        if node.name in self.prev_seen:
            if node.name not in self.new_name_map:
                old = node.name
                node.name = old + str(UniqueNamer.unique_num)
                self.new_name_map[old] = node.name
            else:
                node.name = self.new_name_map[node.name]
        else:
            self.seen.add(node.name)
        node.defn = map(self.visit, node.defn)
        node.params = map(self.visit, node.params)
        return node

    def visit_SymbolRef(self, node):
        if node.name in self.prev_seen:
            if node.name not in self.new_name_map:
                old = node.name
                node.name = old + str(UniqueNamer.unique_num)
                self.new_name_map[old] = node.name
            else:
                node.name = self.new_name_map[node.name]
        else:
            self.seen.add(node.name)
        return node


class FuseFunctions(ast.NodeTransformer):

    def __init__(self):
        super(FuseFunctions, self).__init__()
        self.base_function = None

    def visit_FunctionDecl(self, node):
        if node.inline:
            return node
        if self.base_function is None:
            self.base_function = node
            return node
        self.base_function.defn.extend(node.defn)
        self.base_function.params.extend(node.params)
        return []


class FuseLoops(ast.NodeTransformer):

    def __init__(self):
        super(FuseLoops, self).__init__()
        self.loops_seen = []

    def visit_For(self, node):
        for loop in self.loops_seen:
            if loop.init.right.value == node.init.right.value:
                if loop.test.right.value == node.test.right.value:
                    VarReplacer(
                        loop.init.left.name,
                        node.init.left.name).visit(node)
                    loop.body.extend(node.body)
                    return []
        self.loops_seen.append(node)
        return [StringTemplate("# pragma omp parallel for"), node]


class VarReplacer(ast.NodeTransformer):

    def __init__(self, new, old):
        self.new = new
        self.old = old

    def visit_SymbolRef(self, node):
        if node.name == self.old:
            node.name = self.new
        return node


class BlockBuilder(ast.NodeVisitor):

    def __init__(self, fuser):
        super(BlockBuilder, self).__init__()
        self.fuser = fuser

    def visit_Call(self, node):
        for arg in node.args:
            self.fuser.arg_names.append(arg.id)
        args = [node.func]
        args.extend(node.args)
        append_block = ast.Expression(ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name('self', ast.Load()),
                    attr='fuser',
                    ctx=ast.Load()
                    ),
                attr='append_block',
                ctx=ast.Load()
                ),
            args=args,
            keywords=[],
            starargs=None,
            kwargs=None
            )
            )
        append_block = ast.fix_missing_locations(append_block)
        eval(compile(append_block, '', 'eval'))

# ---------------------------------------------------------------------------
# User code


class Add(ElementwiseArrayOp):

    def apply(a, b):
        return a + b


class Sub(ElementwiseArrayOp):

    def apply(a, b):
        return a - b


def py_add(a, b):
    return a + b


def py_sub(a, b):
    return a - b


class Timer(object):

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.interval = time.clock() - self.start

c_add = Add()
c_sub = Sub()

a = np.ones(2**20, dtype=np.float64)
b = np.ones(2**20, dtype=np.float64)
c = np.ones(2**20, dtype=np.float64)
tmp1 = np.ones(2**20, dtype=np.float64)
actual_d = np.ones(2**20, dtype=np.float64)


class Fuse(Fuser):

    def fuse(self):
        c_add(a, b, tmp1)
        c_add(tmp1, c, actual_d)
Fuse().fuse()
expected_d = py_add(py_add(a, b), c)
np.testing.assert_array_equal(actual_d, expected_d)

tmp2 = np.ones(2**20, dtype=np.float64)
d = np.ones(2**20, dtype=np.float64)
actual_e = np.ones(2**20, dtype=np.float64)


class Fuse(Fuser):

    def fuse(self):
        c_add(a, b, tmp1)
        c_add(tmp1, c, tmp2)
        c_sub(tmp2, d, actual_e)
Fuse().fuse()

expected_e = py_sub(py_add(py_add(a, b), c), d)
np.testing.assert_array_equal(actual_e, expected_e)

tmp3 = np.ones(2**20, dtype=np.float64)
e = np.ones(2**20, dtype=np.float64)
actual_f = np.ones(2**20, dtype=np.float64)


class Fuse(Fuser):

    def fuse(self):
        c_add(a, b, tmp1)
        c_add(tmp1, c, tmp2)
        c_sub(tmp2, d, tmp3)
        c_add(tmp3, e, actual_f)

f = Fuse()
f.fuse()
with Timer() as fused:
    f.fuse()
c_add(a, b, tmp1)
c_add(tmp1, c, tmp2)
c_sub(tmp2, d, tmp3)
c_add(tmp3, e, actual_f)
with Timer() as unfused:
    c_add(a, b, tmp1)
    c_add(tmp1, c, tmp2)
    c_sub(tmp2, d, tmp3)
    c_add(tmp3, e, actual_f)
with Timer() as py:
    expected_f = py_add(py_sub(py_add(py_add(a, b), c), d), e)
print "Fused c time: %.03fs" % fused.interval
print "Unfused c time: %.03fs" % unfused.interval
print "Python time: %.03fs" % py.interval
np.testing.assert_array_equal(actual_f, expected_f)

print("Success.")
