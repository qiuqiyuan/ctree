class BasicBlockNode(object):
    pass


class Symbol(BasicBlockNode):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name


class Constant(BasicBlockNode):
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def __str__(self):
        return str(self._value)


class Param(BasicBlockNode):
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

    def __str__(self):
        return self._name


class Return(BasicBlockNode):
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    def __str__(self):
        return "return {}".format(self._value)


class Assign(BasicBlockNode):
    def __init__(self, target, value):
        self._target = target
        self._value = value

    @property
    def target(self):
        return self._target

    @property
    def value(self):
        return self._value

    def __str__(self):
        return "{} = {}".format(str(self._target), str(self._value))


class FunctionCall(BasicBlockNode):
    def __init__(self, func, args):
        self._func = func
        self._args = args

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args[:]

    def __str__(self):
        return "{}({})".format(str(self._func), ", ".join(map(str, self._args)))
    
            
