import ast
try:
  from pydot import *
except ImportError:
  print("pydot not available.")

from ctree.visitors import NodeVisitor

class Visualizer(NodeVisitor):
  """
  Generates a graphical representation of the AST.
  """

  def __init__(self, name):
    super()
    self.graph = Dot(graph_type='digraph')
    self.graph_name = name + ".png" if name else 'default.png'

  def generate_from(self, node):
    self.visit(node)
    self.graph.write_png(self.graph_name)

  def label_SymbolRef(self, node):
    return "name: %s" % node.name

  def label_Constant(self, node):
    return "value: %s" % node.value

  def label_String(self, node):
    return "value: %s" % node.value

  def label(self, node):
    """
    A string to provide useful information for visualization, debugging, etc.
    This routine will attempt to call a label_XXX routine for class XXX, if
    such a routine exists (much like the visit_XXX routines).
    """
    s = r"%s\n" % type(node).__name__
    labeller = getattr(self, "label_" + type(node).__name__, None)
    if labeller:
      s += labeller(node)
    return s

  def generic_visit(self, node):
    # label this node
    graph_node = Node(self.label(node))
    self.graph.add_node(graph_node)

    for fieldname, child in ast.iter_fields(node):
      if type(child) is list:
        for i, grandchild in enumerate(child):
          grandchild = self.visit(grandchild)
          self.graph.add_edge(Edge(graph_node, grandchild, label="%s[%d]" % (fieldname, i)))
          self.graph.add_edge(Edge(grandchild, graph_node, label="parent", style="dotted"))
      elif isinstance(child, ast.AST):
        child = self.visit(child)
        self.graph.add_edge(Edge(graph_node, child, label=fieldname))
        self.graph.add_edge(Edge(child, graph_node, label="parent", style="dotted"))
    return graph_node

def visualize(node, name=None):
  assert isinstance(node, ast.AST), \
    "visualize expected an instance of ast.AST, got %s." % type(node).__name__
  return Visualizer(name).generate_from(node)
