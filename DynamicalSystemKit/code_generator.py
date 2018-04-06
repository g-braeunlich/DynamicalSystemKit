import types


class CodeGenerator:

    @classmethod
    def build(cls, *args, **kwargs):
        generator = cls(*args, **kwargs)
        return generator.generate()

    def __init__(self):
        pass

    def generate(self):
        pass


class Function(CodeGenerator):

    def __init__(self, name="anonymous",
                 args=(),
                 kwargs=None,
                 code_lines="",
                 environment=None):
        super().__init__()
        self.code_lines = code_lines
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name
        if environment is not None:
            self.environment = environment.copy()
        else:
            self.environment = {}

    def build_code(self):
        args = ", ".join(self.args)
        kwargs = ", ".join(k + "=" + v for k, v in self.kwargs.items())
        signature = ", ".join((args, kwargs))
        return """def {}({}):
{}""".format(self.name, signature, indent(self.code_lines))

    def generate(self):
        exec_dict = {}
        code = self.build_code()
        # pylint: disable=exec-used
        exec(code, self.environment, exec_dict)
        return exec_dict[self.name]


def indent(code_lines):
    return "".join("\n    " + line for line in code_lines)


class Lambda(Function):

    def __init__(self, expression, args=(), environment=None):
        super().__init__(name="Lambda", args=args, code_lines=(
            "return " + expression, ), environment=environment)


class Method(Function):

    def __init__(self, instance, name="anonymous_method", **kwargs):
        super().__init__(name=name, **kwargs)
        self.instance = instance
        self.args = ("self",) + tuple(self.args)

    def generate(self):
        return types.MethodType(super().generate(), self.instance)


class Code(CodeGenerator):
    pass


class ConditionalCodeBlocks(Code):

    def __init__(self, tagged_code_blocks, tags):
        super().__init__()
        self.tagged_code_blocks = tagged_code_blocks
        self.tags = tags

    def generate(self):
        return (block for block,
                tag in self.tagged_code_blocks
                if block and (tag in self.tags or tag is None))


def line_numbered_code_lines(code_lines, start=3):
    return "\n".join("{}:\t{}".format(n, l) for n, l in enumerate(code_lines, start))
