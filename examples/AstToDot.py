"""
Illustrates the DOT-printing functionality by constructing
a small AST and printing its DOT representation.

Usage:
  $ python AstToDot.py > graph.dot

The resulting file can be viewed with a visualizer like Graphiz.
"""

import ctypes as ct

from ctree.nodes.common import *
from ctree.c.nodes import *

from ctree.dotgen import to_dot

def main():
  stmt0 = Assign(SymbolRef('foo'), Constant(123.4))
  stmt1 = FunctionDecl(ct.c_float, SymbolRef("bar"), [SymbolRef("spam", ct.c_int), SymbolRef("eggs", ct.c_long)], [String("baz")])
  tree = File([stmt0, stmt1])
  print (to_dot(tree))

if __name__ == '__main__':
  main()
