import unittest
import numpy as np
import pycl as cl
import ctypes as ct

from ctree.meta.core import meta
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
from ctree.nodes import Project
from ctree.c.nodes import FunctionCall, FunctionDecl, SymbolRef, Constant, \
    Assign, ArrayRef, Add, Ref, CFile
from ctree.templates.nodes import StringTemplate
from ctree.ocl.nodes import OclFile
from ctree.ocl.macros import clSetKernelArg, get_global_id, NULL


class OclFunc(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context)

    def finalize(self, kernel, tree, entry_name, entry_type):
        self.kernel = kernel
        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, A, B):
        a_buf, evt = cl.buffer_from_ndarray(self.queue, A, blocking=False)
        evt.wait()
        b_buf, evt = cl.buffer_from_ndarray(self.queue, B, blocking=False)
        evt.wait()
        C = np.zeros_like(A)
        c_buf, evt = cl.buffer_from_ndarray(self.queue, C, blocking=False)
        evt.wait()
        self._c_function(self.queue, self.kernel, a_buf, b_buf, c_buf)
        _, evt = cl.buffer_to_ndarray(self.queue, c_buf, C)
        evt.wait()
        return C


class OclDoubler(LazySpecializedFunction):
    def args_to_subconfig(self, args):
        A = args[0]
        return tuple(np.ctypeslib.ndpointer(A.dtype, A.ndim, A.shape)
                     for _ in args + (None, ))

    def transform(self, tree, program_config):
        A = program_config[0]
        B = program_config[1]
        C = program_config[2]
        len_A = np.prod(A.__shape__)
        # inner_type = A.__dtype__.type()

        kernel = FunctionDecl(
            None, "kernel",
            params=[SymbolRef("A", A()).set_global(),
                    SymbolRef("B", B()).set_global(),
                    SymbolRef("C", C()).set_global()],
            defn=[Assign(ArrayRef(SymbolRef("C"), get_global_id(0)),
                         Add(SymbolRef('A'), SymbolRef('B')))]
        ).set_kernel()

        file = OclFile("kernel", [kernel])

        control = [
            StringTemplate("""
                #ifdef __APPLE__
                #include <OpenCL/opencl.h>
                #else
                #include <CL/cl.h>
                #endif
            """),
            FunctionDecl(
                None, params=[SymbolRef('queue', cl.cl_command_queue()),
                              SymbolRef('kernel', cl.cl_kernel()),
                              SymbolRef('a', cl.cl_mem()),
                              SymbolRef('b', cl.cl_mem()),
                              SymbolRef('c', cl.cl_mem())],
                defn=[
                    Assign(SymbolRef('global', ct.c_ulong()), Constant(len_A)),
                    Assign(SymbolRef('local', ct.c_ulong()), Constant(32)),
                    clSetKernelArg('kernel', 0, ct.sizeof(cl.cl_mem), 'a'),
                    clSetKernelArg('kernel', 1, ct.sizeof(cl.cl_mem), 'b'),
                    clSetKernelArg('kernel', 2, ct.sizeof(cl.cl_mem), 'c'),
                    FunctionCall(SymbolRef('clSetKernelArg'),
                                 [SymbolRef('queue'), SymbolRef('kernel'),
                                  Constant(1), Constant(0),
                                  Ref(SymbolRef('global')),
                                  Ref(SymbolRef('local')), Constant(0),
                                  NULL(), NULL()]),
                    FunctionCall(SymbolRef('clFinish'), [SymbolRef('queue')])
                ])
        ]

        proj = Project([file, CFile('control', control)])
        fn = OclFunc()
        program = cl.clCreateProgramWithSource(fn.context,
                                               kernel.codegen()).build()
        ptr = program['kernel']
        entry_type = ct.CFUNCTYPE(None, cl.cl_command_queue, cl.cl_kernel,
                                  cl.cl_mem, cl.cl_mem, cl.cl_mem)
        return fn.finalize(ptr, proj, "control", entry_type)


class TestMetaDecorator(unittest.TestCase):
    def test_simple(self):
        @meta
        def func(a):
            return a + 3

        self.assertEqual(func(3), 6)
