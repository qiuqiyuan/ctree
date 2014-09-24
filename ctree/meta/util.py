__author__ = 'leonardtruong'


class SymbolTable(dict):
    """
    Wrapper around a Python Dictionary to support lazy lookup of
    symbols in the current frame.
    """
    def __init__(self, env, stack):
        super(SymbolTable, self).__init__()
        self._env = env
        self._stack = stack

    def __getitem__(self, symbol):
        try:
            return self._env[symbol]
        except KeyError:
            # Check stack for symbol
            for frame in self._stack:
                if symbol in frame[0].f_locals:
                    # Cache it
                    self._env[symbol] = frame[0].f_locals[symbol]
                    return self._env[symbol]
            # TODO: Make this a more meaningful exception
            raise KeyError(symbol)

    def __setitem__(self, symbol, item):
        self._env[symbol] = item

    def __contains__(self, symbol):
        try:
            self[symbol]
            return True
        except KeyError:
            return False
