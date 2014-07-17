try:
    __IPYTHON__
    import sys
    sys.path.append('/Users/leonardtruong/aspire_projects/pycl')
    sys.path.append('/Users/leonardtruong/aspire_projects/stencil_code')
    # %load_ext autoreload
    # %autoreload 2
except:
    pass

from stencil_code.stencil_kernel import StencilKernel
from stencil_code.stencil_grid import StencilGrid
from ctree.frontend import get_ast
from ctree.c.nodes import *
from ctree.templates.nodes import StringTemplate
import ctree

import ast
import pycl as cl
import ctypes as ct
import numpy as np

import time

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


###########################################################
# Create a simple kernel to fuse

class Kernel(StencilKernel):
    def kernel(self, in_grid, out_grid):
        for x in out_grid.interior_points():
            for y in in_grid.neighbors(x, 1):
                out_grid[x] += in_grid[y]

width = 2**12 + 2
stencil_kernel = Kernel(backend='ocl')
b = StencilGrid([width, width])
b.ghost_depth = 1
a = StencilGrid([width, width])
a.ghost_depth = 1
c = StencilGrid([width, width])
c.ghost_depth = 1

# a.data = np.ones([width, width])
a.data = np.ones([width, width], dtype=np.float32)
# for x in a.interior_points():
#     a[x] = 1.0

###########################################################

###########################################################
# Define our fusion operation

def fuse(self):
    stencil_kernel.kernel(a, b)
    stencil_kernel.kernel(b, c)

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

###########################################################
# Do the fusion

tree = get_ast(fuse)
SemanticModelBuilder().visit(tree)
# Remove fuse function def node
tree.body = tree.body[0].body

num_fusions = len(tree.body)

for t in tree.body:
    t.backend_transform()
# ctree.browser_show_ast(tree, 'tmp.png')
kernel = tree.body[0].function_decl.files[0].body[0]
kernel.defn.extend(tree.body[1].function_decl.files[0].body[0].defn[3:])

# Update our global indices to account for fusion padding
kernel.defn[2].right.left.left.right.value *= num_fusions
kernel.defn[2].right.right.right.value *= num_fusions

# TODO: This needs to check validity of fusion operation, i.e. do source and
# sinks match.
for i in range(1, num_fusions):
    kernel.params.append(SymbolRef('block%d' % i, ct.POINTER(ct.c_float)(), _local=True))

# Update expanded block sizes
# TODO: For some reason these constants are linked, updating one updates both
kernel.defn[4].right.right.right.value *= num_fusions
kernel.defn[6].body[0].right.right.right.value *= num_fusions

# Move the first stencil operation into the shared memory load block of the
# second stencil.  The operation is modified to output into local memory instead
# of global memory, as well as updating it's indexing scheme into the original
# shared memory block since our padding has changed.
stencil_op = kernel.defn[10:14]
del kernel.defn[10:14]
for line in stencil_op:
    line.target.left = SymbolRef('block1')
    line.target.right = SymbolRef('tid')
    line.value.right.func = 'local_array_macro2'
    for arg in line.value.right.args:
        arg.left = SymbolRef('i_' + arg.left.name[-1])
stencil_op.insert(0, Assign(SymbolRef('block1[tid]'), Constant(0)))

# Remove the old shared memory load
del kernel.defn[13].body[-1]

# Insert the modified stencil operations.
kernel.defn[13].body.extend(stencil_op)

# Remove the redecleration of variables
# TODO: This should only redefine block_size as the others are unchanged.
for decl in kernel.defn[10:13]:
    decl.left.type = None

# Rename our final stencil operatiosn to use the new shared memory block from
# the first stencil
for op in kernel.defn[17:]:
    op.value.left.name = 'block1'
del kernel.defn[8:10]
kernel.defn.insert(1, StringTemplate("#define local_array_macro2(d0, d1) ((d1 + 1) * (get_local_size(0) + 4) + (d0 + 1))"))

# print(kernel)

###########################################################

out_grid = StencilGrid([width, width])
out_grid.ghost_depth = 1
in_grid = StencilGrid([width, width])
in_grid.ghost_depth = 1

# for x in in_grid.interior_points():
#     in_grid[x] = 1.0
in_grid.data = np.ones([width, width], dtype=np.float32)
with Timer() as t:
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
    evt = cl.clEnqueueNDRangeKernel(queue, kernel, (width - 2, width - 2), (local, local))
    evt.wait()

    ary, evt = cl.buffer_to_ndarray(queue, out_buf, out_grid.data)
    evt.wait()
print("Fused time: %f" % t.interval)

# print(ary[2:-2, 2:-2])

with Timer() as u:
    stencil_kernel.kernel(a, b)
    stencil_kernel.kernel(b, c)
print("Unfused time: %f" % u.interval)

np.testing.assert_array_equal(ary[2:-2, 2:-2], c[2:-2, 2:-2])
print('PASSED')
