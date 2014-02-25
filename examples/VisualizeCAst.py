"""
Illustrates the AST visualization functionality by constructing
a small AST and generating a graphical representation: c_ast.png.

Dependencies: pydot
  $ pip install pydot2

Usage:
  $ python3 AstToDot.py
"""

from ctree.nodes.c import *

def main():
  stmt0 = Assign(SymbolRef('foo'), Constant(123.4))
  stmt1 = FunctionDecl(Float(), SymbolRef("bar"), [Int(), Long()], [String("baz")])
  tree = File([stmt0, stmt1])
  print (tree.visualize("c_ast"))

if __name__ == '__main__':
  main()