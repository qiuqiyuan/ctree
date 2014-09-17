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


class Op(BasicBlockNode):
    def __init__(self, op, args):
        self._op = op
        self._args = args

    @property
    def op(self):
        return self._op

    @property
    def args(self):
        return self._args[:]

    def __str__(self):
        return "{}({})".format(str(self._op), ", ".join(map(str, self._args)))


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
