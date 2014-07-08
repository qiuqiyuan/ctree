__author__ = 'leonardtruong'
  os.getcwd()

from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
from ctree.frontend import get_ast
import ctree

import ast
import pycl as cl
import ctypes as ct
import numpy as np


class Fusable(object):
    def __init__(self):
        self.body = self.fuse
        self.fuse = self.shadow_fuse

    def one_d_fuse(self):
        tree = get_ast(self.body)
        SemanticModelBuilder().visit(tree)

        # Remove fuse functiondef node
        tree.body = tree.body[0].body
        # Fuse projects and remove python module and fuse functiondef nodes
        # for proj in tree.body[0].body[1:]:
        #     tree.body[0].body[0].files.extend(proj.files)
        # tree = tree.body[0].body[0]

        tree.body[0].backend_transform()
        tree.body[1].backend_transform()
        kernel = tree.body[0].function_decl.files[0].body[0]

        kernel = StringTemplate("""
        __kernel void stencil_kernel(__global const float* in_grid, __global float* out_grid, __local float* block) {
            int id0 = get_global_id(0) + 2;
            int global_index = id0;
            #define local_array_macro(d0) (d0)
            #define global_array_macro(d0) (d0)
            for (int d0 = get_local_id(0); d0 < get_local_size(0) + 4; d0 += get_local_size(0)) {
            block[local_array_macro(d0)] = in_grid[global_array_macro(d0 + get_group_id(0) * get_local_size(0))];
            }
            float tmp1 = 0.0;
            float tmp2 = 0.0;

            barrier(CLK_LOCAL_MEM_FENCE);
            int local_id0 = get_local_id(0) + 2;

            if (get_local_id(0) == 0) {
                tmp1 = block[local_id0 - 2];
                tmp1 += block[local_id0];
            } else {
                tmp1 = block[local_id0];
                tmp1 += block[local_id0 + 2];
            }
            tmp2 = block[local_id0 - 1];
            tmp2 += block[local_id0 + 1];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) == 0) {
            block[local_id0 - 1] = tmp1;
            } else {
            block[local_id0 + 1] = tmp1;
            }
            block[local_id0] = tmp2;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) == 0 && get_group_id(0) == 0) {
            out_grid[global_index - 1] += block[local_id0 - 2];
            out_grid[global_index - 1] += block[local_id0];
            }
            if (get_local_id(0) == 1 && get_group_id(0) == get_num_groups(0) - 1) {
            out_grid[global_index + 1] += block[local_id0 + 2];
            out_grid[global_index + 1] += block[local_id0];
            }
            out_grid[global_index] += block[local_id0 - 1];
            out_grid[global_index] += block[local_id0 + 1];
        }
        """)
        print(kernel.codegen())
        gpus = cl.clGetDeviceIDs(device_type=cl.cl_device_type.CL_DEVICE_TYPE_GPU)
        context = cl.clCreateContext([gpus[1]])
        queue = cl.clCreateCommandQueue(context)

        width = 68
        out_grid = StencilGrid([width])
        out_grid.ghost_depth = 1
        in_grid = StencilGrid([width])
        in_grid.ghost_depth = 1

        for x in in_grid.interior_points():
            in_grid[x] = 1.0

        local = 2
        program = cl.clCreateProgramWithSource(context, kernel.codegen()).build()
        kernel = program['stencil_kernel']

        in_buf, evt = cl.buffer_from_ndarray(queue, in_grid.data)
        print(in_grid.data.dtype)
        evt.wait()
        kernel.setarg(0, in_buf, ct.sizeof(cl.cl_mem))

        out_buf, evt = cl.buffer_from_ndarray(queue, out_grid.data)
        evt.wait()
        kernel.setarg(1, out_buf, ct.sizeof(cl.cl_mem))

        local_mem_size = ct.sizeof(ct.c_float) * (local + 2)
        local_mem = cl.localmem(local_mem_size)
        kernel.setarg(2, local_mem, local_mem_size)

        evt = cl.clEnqueueNDRangeKernel(queue, kernel, (width - 4,), (local,))
        evt.wait()

        ary, evt = cl.buffer_to_ndarray(queue, out_buf, out_grid.data)
        evt.wait()
        return ary

    def shadow_fuse(self):
        tree = get_ast(self.body)
        SemanticModelBuilder().visit(tree)

        # Remove fuse functiondef node
        tree.body = tree.body[0].body
        # Fuse projects and remove python module and fuse functiondef nodes
        # for proj in tree.body[0].body[1:]:
        #     tree.body[0].body[0].files.extend(proj.files)
        # tree = tree.body[0].body[0]

        tree.body[0].backend_transform()
        tree.body[1].backend_transform()
        exit(1)
        kernel = StringTemplate("""
        __kernel void stencil_kernel(__global const float* in_grid, __global
        float* out_grid, __local float* block, __local float* block2) {
            if (get_global_id(0) > 1024 || get_global_id(1) > 1024) return;
            int id0 = get_global_id(0) + 2;
            int global_index = id0;
            #define local_array_macro(d0, d1) ((d1) * (get_local_size(0) + 4) + (d0))
            #define local_array_macro2(d0, d1) ((d1) * (get_local_size(0) + 2) + (d0))
            // for (int d0 = get_local_id(0); d0 < get_local_size(0) + 4; d0 += get_local_size(0)) {
            //     for (int d1 = get_local_id(1); d1 < get_local_size(1) + 4; d1 += get_local_size(1)) {
            //         block[local_array_macro(d0, d1)] = in_grid[global_array_macro(d0 + get_group_id(0) * get_local_size(0), d1 + get_group_id(1) * get_local_size(1))];
            //     }
            // }
            #define global_array_macro(d0, d1) ((d1) * (get_global_size(0) + 2) + (d0))

            for (int tid = get_local_id(1) * get_local_size(0) + get_local_id(0); tid < (get_local_size(0) + 4) * (get_local_size(1) + 4); tid += get_local_size(0) * get_local_size(1)) {
                int local_x = tid % (get_local_size(0) + 4);
                int glb_x = local_x + get_group_id(0) * (get_local_size(0));
                int glb_y = (tid - local_x) / (get_local_size(0) + 4) + get_group_id(1) * (get_local_size(1));
                block[tid] = in_grid[global_array_macro(glb_x, glb_y)];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int tid = get_local_id(1) * get_local_size(0) + get_local_id(0); tid < (get_local_size(0) + 2) * (get_local_size(1) + 2); tid += get_local_size(0) * get_local_size(1)) {
                int local_x = tid % (get_local_size(0) + 2);
                int local_y = (tid - local_x) / (get_local_size(0) + 2);
                block2[tid] = 0;
                block2[tid] += block[local_array_macro(local_x + 1, local_y)];
                block2[tid] += block[local_array_macro(local_x + 1, local_y + 2)];
                block2[tid] += block[local_array_macro(local_x, local_y + 1)];
                block2[tid] += block[local_array_macro(local_x + 2, local_y + 1)];
            //for (int d0 = get_local_id(0); d0 < get_local_size(0) + 2; d0 += get_local_size(0)) {
            //    for (int d1 = get_local_id(1); d1 < get_local_size(1) + 2; d1 += get_local_size(1)) {
            //        float tmp = 0.0;
            //        tmp += block[local_array_macro(d0 + 1, d1)];
            //        tmp += block[local_array_macro(d0 + 1, d1 + 2)];
            //        tmp += block[local_array_macro(d0, d1 + 1)];
            //        tmp += block[local_array_macro(d0 + 2, d1 + 1)];
            //        block2[local_array_macro2(d0, d1)] = tmp;
            //    }
            //}
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            out_grid[global_array_macro(get_global_id(0) + 2, get_global_id(1) + 2)] += block2[local_array_macro2(get_local_id(0) + 1, get_local_id(1))];
            out_grid[global_array_macro(get_global_id(0) + 2, get_global_id(1) + 2)] += block2[local_array_macro2(get_local_id(0) + 1, get_local_id(1) + 2)];
            out_grid[global_array_macro(get_global_id(0) + 2, get_global_id(1) + 2)] += block2[local_array_macro2(get_local_id(0), get_local_id(1) + 1)];
            out_grid[global_array_macro(get_global_id(0) + 2, get_global_id(1) + 2)] += block2[local_array_macro2(get_local_id(0) + 2, get_local_id(1) + 1)];
        }
        """)

        width = 2**10 + 2
        out_grid = StencilGrid([width, width])
        out_grid.ghost_depth = 1
        in_grid = StencilGrid([width, width])
        in_grid.ghost_depth = 1

        for x in in_grid.interior_points():
            in_grid[x] = 1.0

        gpus = cl.clGetDeviceIDs(device_type=cl.cl_device_type.CL_DEVICE_TYPE_GPU)
        context = cl.clCreateContext([gpus[1]])
        queue = cl.clCreateCommandQueue(context)
        local = 32
        program = cl.clCreateProgramWithSource(context, kernel.codegen()).build()
        kernel = program['stencil_kernel']
        events = []

        in_buf, evt = cl.buffer_from_ndarray(queue, in_grid.data)
        kernel.setarg(0, in_buf, ct.sizeof(cl.cl_mem))
        events.append(evt)

        out_buf, evt = cl.buffer_from_ndarray(queue, out_grid.data)
        kernel.setarg(1, out_buf, ct.sizeof(cl.cl_mem))
        events.append(evt)

        block_size = ct.sizeof(ct.c_float) * (local + 4) * (local + 4)
        block = cl.localmem(block_size)
        kernel.setarg(2, block, block_size)

        block2_size = ct.sizeof(ct.c_float) * (local + 2) * (local + 2)
        block2 = cl.localmem(block2_size)
        kernel.setarg(3, block2, block2_size)

        cl.clWaitForEvents(*events)
        total = 0
        for i in range(10):
            with Timer() as t:
                evt = cl.clEnqueueNDRangeKernel(queue, kernel, (width - 2, width - 2), (local, local))
                evt.wait()
            total += t.interval
        print(total / 10)

        ary, evt = cl.buffer_to_ndarray(queue, out_buf, out_grid.data)
        evt.wait()
        return ary


class SemanticModelBuilder(ast.NodeTransformer):
    def get_node(self, node):
        self.current_node = node

    def visit_Expr(self, node):
        node.value.func.attr = 'get_semantic_node'
        expr = ast.Expression(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name('self', ast.Load()),
                    attr='get_node',
                    ctx=ast.Load()
                ),
                args=[node.value],
                keywords=[]
            )
        )
        expr = ast.fix_missing_locations(expr)
        exec(compile(expr, filename='', mode='eval'))
        return self.current_node


class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

width = 2**10 + 2
kernel = Kernel()
b = StencilGrid([width, width])
b.ghost_depth = 1
a = StencilGrid([width, width])
a.ghost_depth = 1
c = StencilGrid([width, width])
c.ghost_depth = 1

for x in a.interior_points():
    a[x] = 1.0

stencil_kernel = Kernel(backend='ocl')


class Fuse(Fusable):
    def fuse(self):
        stencil_kernel.kernel(a, b)
        stencil_kernel.kernel(b, c)

actual = Fuse().fuse()
exit(0)
with Timer() as t:
    stencil_kernel.kernel(a, b)
    stencil_kernel.kernel(b, c)
print t.interval

np.testing.assert_array_equal(actual[2:-2, 2:-2], c.data[2:-2, 2:-2])
print 'SUCCESS'
