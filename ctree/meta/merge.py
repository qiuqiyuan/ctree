__author__ = 'leonardtruong'

import ast
import sys
from ctree.ocl import get_context_and_queue_from_devices
import pycl as cl
import ctypes as ct
import numpy as np
from ctree.nodes import Project
from .util import get_unique_func_name, UniqueNamer, find_entry_point, \
    SymbolReplacer

from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction


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
                program = cl.clCreateProgramWithSource(
                    self.context, kernel[1].codegen()).build()
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
                           self.__entry_type, self.__kernels,
                           self.__output_indexes)


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
            param_map[statement.targets[0].id] = \
                uniquifier.seen[statement.targets[0].id]
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
    merged_name = get_unique_func_name(env)
    env[merged_name] = MergedSpecializedFunction(Project(files),
                                                 merged_entry.name.name,
                                                 merged_entry_type,
                                                 merged_kernels,
                                                 output_indexes)
    value = ast.Call(ast.Name(merged_name, ast.Load()), args, [], None, None)
    return ast.Assign(targets, value)


class MergeableInfo(object):
    def __init__(self, proj=None, entry_point=None, entry_type=None,
                 kernels=None):
        self.proj = proj
        self.entry_point = entry_point
        self.entry_type = entry_type
        self.kernels = kernels
