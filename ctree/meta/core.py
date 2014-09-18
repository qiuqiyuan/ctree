from ctree.frontend import get_ast
from .basic_blocks import get_basic_block


class MetaSpecializer(object):
    """docstring for MetaSpecializer"""
    def __init__(self, func):
        super(MetaSpecializer, self).__init__()
        self._func = func
        self._original_ast = get_ast(func)

    def __call__(self, *args, **kwargs):
        basic_block = get_basic_block(self._original_ast)
        return basic_block

meta = MetaSpecializer
