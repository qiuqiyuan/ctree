def fib(n):
  """Arbitrary python code here."""
  prev, curr = 0, 1
  for i in range(1, n):
  	prev, curr = curr, prev + curr
  return curr

import ast, inspect
from ctree.dotgen import *
program_txt = inspect.getsource(fib)
py_ast = ast.parse(program_txt)
print( to_dot(py_ast) )