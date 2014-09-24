import ast
from functools import reduce
import sys
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.ocl import get_context_and_queue_from_devices
import pycl as cl
import ctypes as ct
import numpy as np
from ctree.nodes import Project


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
    elif isinstance(item, NonComposableBlock):
        tab = "\n" + "".join([" " for _ in range(tab + 2)])
        return "NonComposableBlock:{}{}".format(tab, tab.join(
            map(str_dump, item.statements)))
    elif isinstance(item, ast.arguments):
        if sys.version_info >= (3, 0):
            return ", ".join(arg.arg for arg in item.args)
        else:
            return ", ".join(arg.id for arg in item.args)
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

    def __iter__(self):
        return iter(self.body)

    def __repr__(self):
        return """
BasicBlock
  Name: {name}
  Params: {params}
  Body:
    {body}
""".format(name=self.name,
           params=str_dump(self.params),
           body="\n    ".join(map(lambda x: str_dump(x, 4), self.body)))


def is_composable(statement, env):
    return isinstance(statement, ast.Assign) and \
        isinstance(statement.value, ast.Call) and \
        isinstance(eval_in_env(env, statement.value.func),
                   LazySpecializedFunction)


def separate_composable_blocks(basic_block, env):
    # TODO: This is a pretty convoluted function, simplify it to a
    # reduction across the block
    statements = []
    for statement in basic_block.body:
        if is_composable(statement, env):
            if len(statements) > 0 and \
               isinstance(statements[-1], ComposableBlock):
                statements[-1].add_statement(statement)
            else:
                statements.append(ComposableBlock([statement]))
        else:
            if len(statements) > 0 and \
               isinstance(statements[-1], NonComposableBlock):
                statements[-1].add_statement(statement)
            else:
                statements.append(NonComposableBlock([statement]))

    return BasicBlock(basic_block.name, basic_block.params,
                      statements)


class SubBlock(object):
    def __init__(self, statements):
        super(SubBlock, self).__init__()
        self.statements = statements
        self.live_ins = set()
        self.live_outs = set()

    def add_statement(self, item):
        self.statements.append(item)

    def __iter__(self):
        return iter(self.statements)

    def __getitem__(self, item):
        return self.statements[item]


class ComposableBlock(SubBlock):
    """docstring for ComposableBlock"""
    pass


class NonComposableBlock(SubBlock):
    """docstring for NonComposableBlock"""
    pass


def decompose(expr):
    def gen_tmp():
        gen_tmp.tmp += 1
        return "_t{}".format(gen_tmp.tmp)
    gen_tmp.tmp = -1

    def visit(expr, curr_target=None):
        if isinstance(expr, ast.Return):
            if isinstance(expr.value, ast.Name):
                body = (expr, )
            else:
                tmp = gen_tmp()
                body = visit(expr.value, ast.Name(tmp, ast.Store()))
                body += (ast.Return(ast.Name(tmp, ast.Load())), )
        elif isinstance(expr, ast.Name):
            return expr
        elif isinstance(expr, ast.BinOp):
            body = ()
            operands = []

            if isinstance(expr.left, ast.Num):
                body += (ast.Assign([curr_target], expr), )
            else:
                for operand in [expr.left, expr.right]:
                    if isinstance(operand, (ast.Name, ast.Num)):
                        operands += (operand, )
                    else:
                        tmp = gen_tmp()
                        body += visit(operand,
                                      ast.Name(tmp, ast.Store()))
                        operands.append(ast.Name(tmp, ast.Load()))
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
                operands.pop(0)
                body += (ast.Assign([curr_target],
                                    ast.Call(op, operands, [], None, None)), )
        elif isinstance(expr, ast.Assign):
            target = expr.targets[0]
            body = visit(expr.value, target)
        elif isinstance(expr, ast.Call):
            body = ()
            args = []
            for arg in expr.args:
                val = visit(arg)
                if isinstance(val, tuple):
                    tmp = gen_tmp()
                    val = visit(arg, ast.Name(tmp, ast.Store))
                    body += val
                    args.append(ast.Name(tmp, ast.Load()))
                elif isinstance(val, (ast.Name, ast.Num)):
                    args.append(val)
                else:
                    raise Exception("Call argument returned\
                                     unsupported type {}".format(type(val)))
            if curr_target is not None:
                body += (ast.Assign(
                    [curr_target],
                    ast.Call(visit(expr.func), args, [], None, None)
                ), )
            else:
                body += (ast.Call(visit(expr.func), args), )
        else:
            raise Exception("Unsupported expression {}".format(expr))
        return body
    return visit(expr)


def get_basic_block(module):
    func = module.body[0]
    params = func.args
    body = map(decompose, func.body)
    body = reduce(lambda x, y: x + y, body, ())
    return BasicBlock(func.name, params, body)


def get_unique_name(env):
    cnt = 0
    name = "_merged_f0"
    while name in env:
        cnt += 1
        name = "merged_f{}".format(cnt)
    return name


class UniqueNamer(ast.NodeTransformer):
    curr = -1

    def __init__(self):
        self.seen = {}

    def gen_tmp(self):
        UniqueNamer.curr += 1
        return "_f{}".format(UniqueNamer.curr)

    def visit_FunctionCall(self, node):
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_SymbolRef(self, node):
        if node.name == 'NULL':
            return node
        if node.name not in self.seen:
            self.seen[node.name] = self.gen_tmp()
        node.name = self.seen[node.name]
        return node


class EntryPointFinder(ast.NodeVisitor):
    def __init__(self, entry_name):
        self.entry_name = entry_name
        self.entry_point = None

    def visit_FunctionDecl(self, node):
        if node.name.name == self.entry_name:
            self.entry_point = node


def find_entry_point(entry_name, tree):
    finder = EntryPointFinder(entry_name)
    finder.visit(tree)
    if not finder.entry_point:
        raise Exception("Could not find entry point {}".format(entry_name))
    return finder.entry_point


class SymbolReplacer(ast.NodeTransformer):
    def __init__(self, old, new):
        self._old = old
        self._new = new

    def visit_SymbolRef(self, node):
        if node.name == self._old:
            node.name = self._new
        return node


class ConcreteMerged(ConcreteSpecializedFunction):
    def __init__(self):
        devices = cl.clGetDeviceIDs()
        # Default to last device for now
        # TODO: Allow settable devices via params or env variables
        self.context, self.queue = get_context_and_queue_from_devices(
            [devices[-1]])

    def finalize(self, proj, entry_name, entry_type, kernels, outputs):
        self.__entry_type = entry_type
        self._c_function = self._compile(entry_name, proj,
                                         ct.CFUNCTYPE(*entry_type))
        self.__kernels = kernels
        self.__outputs = outputs
        return self

    def __call__(self, *args, **kwargs):
        processed = []
        events = []
        kernel_index = 0
        arg_index = 0
        outputs = []
        for index, argtype in enumerate(self.__entry_type[1:]):
            if argtype is cl.cl_command_queue:
                processed.append(self.queue)
            elif argtype is cl.cl_kernel:
                kernel = self.__kernels[kernel_index]
                program = cl.clCreateProgramWithSource(self.context,
                                                       kernel[1].codegen()).build()
                processed.append(program[kernel[0]])
            elif index + 1 in self.__outputs:
                buf, evt = cl.buffer_from_ndarray(self.queue,
                                                  np.zeros_like(args[0]),
                                                  blocking=False)
                processed.append(buf)
                outputs.append(buf)
                events.append(evt)
            elif isinstance(args[arg_index], np.ndarray) and \
                    argtype is cl.cl_mem:
                buf, evt = cl.buffer_from_ndarray(self.queue, args[arg_index],
                                                  blocking=False)
                processed.append(buf)
                events.append(evt)
                arg_index += 1
        cl.clWaitForEvents(*events)
        self._c_function(*processed)
        buf, evt = cl.buffer_to_ndarray(self.queue, outputs[-1], like=args[0],
                                    blocking=True)
        evt.wait()
        return buf


class MergedSpecializedFunction(LazySpecializedFunction):
    def __init__(self, tree, entry_name, entry_type, kernels, output_indexes):
        super(MergedSpecializedFunction, self).__init__(None)
        self.__original_tree = tree
        self.__entry_name = entry_name
        self.__entry_type = entry_type
        self.__kernels = kernels
        self.__output_indexes = output_indexes

    def transform(self, tree, program_config):
        fn = ConcreteMerged()
        return fn.finalize(self.__original_tree, self.__entry_name,
                           self.__entry_type, self.__kernels, self.__output_indexes)


def replace_symbol_in_tree(tree, old, new):
    replacer = SymbolReplacer(old, new)
    for statement in tree.defn:
        replacer.visit(statement)
    return tree


def merge_entry_points(composable_block, env):
    args = []
    merged_entry_type = [None]
    entry_points = []
    param_map = {}
    seen_args = set()
    files = []
    merged_kernels = []
    output_indexes = []
    for statement in composable_block.statements:
        for arg in statement.value.args:
            if sys.version_info > (3, 0):
                if arg.id in composable_block.live_ins and \
                   arg.id not in seen_args:
                    seen_args.add(arg.id)
                    args.append(arg)
            else:
                if arg.arg in composable_block.live_ins and \
                   arg.arg not in seen_args:
                    seen_args.add(arg.arg)
                    args.append(arg)
        specializer = env[statement.value.func.id]
        program_cfg = specializer.args_to_subconfig(
            tuple(env[arg.id] for arg in statement.value.args))
        placeholder_output = specializer.get_placeholder_output(
            tuple(env[arg.id] for arg in statement.value.args))
        mergeable_info = specializer.get_mergeable_info(
            tuple(env[arg.id] for arg in statement.value.args))
        env[statement.targets[0].id] = placeholder_output
        proj = mergeable_info.proj
        entry_point = mergeable_info.entry_point
        entry_type = mergeable_info.entry_type
        kernels = mergeable_info.kernels
        files.extend(proj.files)
        uniquifier = UniqueNamer()
        uniquifier.visit(proj)
        unique_entry = uniquifier.seen[entry_point]
        if statement.targets[0].id in uniquifier.seen:
            param_map[statement.targets[0].id] = uniquifier.seen[statement.targets[0].id]
        merged_kernels.extend((uniquifier.seen[kernel[0]], kernel[1])
                              for kernel in kernels)
        entry_point = find_entry_point(unique_entry, proj)
        entry_points.append(entry_point)
        to_remove_symbols = set()
        to_remove_types = set()
        for index, arg in enumerate(statement.value.args):
            if arg.id in param_map:
                param = entry_point.params[index + 2].name
                to_remove_symbols.add(param)
                to_remove_types.add(index + 3)
                replace_symbol_in_tree(entry_point, param, param_map[arg.id])
            else:
                param_map[arg.id] = entry_point.params[index + 2].name
        entry_point.params = [p for p in entry_point.params
                              if p.name not in to_remove_symbols]
        entry_type = [type for index, type in enumerate(entry_type)
                      if index not in to_remove_types]
        merged_entry_type.extend(entry_type[1:])
        output_indexes.append(len(merged_entry_type) - 1)
        # entry_points.append(find_entry_point(entry_point, proj))

    merged_entry = entry_points.pop(0)
    for point in entry_points:
        merged_entry.params.extend(point.params)
        merged_entry.defn.extend(point.defn)
        point.delete()

    targets = [ast.Name(id, ast.Store()) for id in composable_block.live_outs]
    merged_name = get_unique_name(env)
    env[merged_name] = MergedSpecializedFunction(Project(files),
                                                 merged_entry.name.name,
                                                 merged_entry_type,
                                                 merged_kernels, output_indexes)
    value = ast.Call(ast.Name(merged_name, ast.Load()), args, [], None, None)
    return ast.Assign(targets, value)


def process_composable_blocks(basic_block, env):
    body = []
    for sub_block in basic_block:
        if isinstance(sub_block, ComposableBlock):
            body.append(merge_entry_points(sub_block, env))
        else:
            body.extend(sub_block.statements)
    return BasicBlock(basic_block.name, basic_block.params, body)


class MergeableInfo(object):
    def __init__(self, proj=None, entry_point=None, entry_type=None,
                  kernels=None):
        self.proj = proj
        self.entry_point = entry_point
        self.entry_type = entry_type
        self.kernels = kernels