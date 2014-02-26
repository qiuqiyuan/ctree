import ast, tkinter
try:
  from pydot import *
except ImportError:
  print("pydot not available.")

from ctree.visitors import NodeVisitor

class Visualizer(NodeVisitor):
  """
  Generates a graphical representation of the AST.
  """

  def __init__(self):
    super()
    self.graph = Dot(graph_type='digraph')

  def generate_from(self, node):
    self.visit(node)
    self.graph.write_gif('tmp.gif')
    root = tkinter.Tk()
    self.photo = tkinter.PhotoImage(file="./tmp.gif")
    canvas = tkinter.Canvas(root, width=self.photo.width(), height=self.photo.height())
    canvas.grid(row = 0, column = 0)
    canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
    root.mainloop()
    import os
    os.remove('./tmp.gif')

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
    graph_node = Node(str(id(node)), label=self.label(node))
    self.graph.add_node(graph_node)

    if hasattr(node, 'parent') and node.parent != None:
      # PyDot returns a list of nodes with the same name.
      # TODO: Can we assume there will only be one node with a certain name?
      parent = self.graph.get_node(str(id(node.parent)))[0]
      self.graph.add_edge(Edge(graph_node, parent, label="parent",
                               style="dotted"))

    for fieldname, child in ast.iter_fields(node):
      if type(child) is list:
        for i, grandchild in enumerate(child):
          self.graph.add_edge(Edge(graph_node, self.visit(grandchild),
                                   label="%s[%d]" % (fieldname, i)))
      elif isinstance(child, ast.AST):
        self.graph.add_edge(Edge(graph_node, self.visit(child),
                                 label=fieldname))
    return graph_node

def visualize(node, name=None):
  assert isinstance(node, ast.AST), \
    "visualize expected an instance of ast.AST, got %s." % type(node).__name__
  return Visualizer(name).generate_from(node)
