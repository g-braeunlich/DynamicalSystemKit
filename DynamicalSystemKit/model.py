import inspect
import warnings
from collections import OrderedDict
import numpy
import scipy

from . import code_generator, numerics, model_json
from .utils import structured_data, depcalc, dct
from .utils.attr import getattr_recursive

# pylint: disable=too-few-public-methods

# TODO: dt_ctrl < dt


class QuantityMeta(type):
    def __call__(cls, *args, name=None, **kwargs):
        unbound = type.__call__(cls, *args, **kwargs)
        if name is not None:
            return unbound.resource(None, name)
        return unbound


class Quantity:
    """ Base class for quantities (state function, control quantity) """

    def __init__(self, n=1):
        self.n = n  #: Dimension of the quantities vector space

    def resource(self, *args, **kwargs):
        return self.Resource(self, *args, **kwargs)

    class Resource:
        def __init__(self, base, parent, alias):
            self.base = base
            self.parent = parent
            self.alias = alias

        def register(self, ode_builder):
            pass

        def expression(self, ode_builder, slc_prefix=""):
            pass

        def dependencies(self):
            return frozenset()

        def computations(self):
            return frozenset()

        def __getitem__(self, slc):
            return ForeignQuantity().resource(alias=self.alias, parent=self.parent,
                                              target=self, sub_slice=slc)

        def __repr__(self):
            parent_repr = self.parent and (repr(self.parent), )
            alias = self.alias or object.__repr__(self)
            return ".".join(((parent_repr or ()) + (alias,)))


class ConstantQuantity(Quantity, metaclass=QuantityMeta):
    def __init__(self, value):
        self.value = value
        shp = numpy.shape(value)
        if shp:
            [n] = shp
        else:
            n = 1
        super().__init__(n=n)

    class Resource(Quantity.Resource):
        def expression(self, ode_builder, slc_prefix=""):
            return str(self.base.value)


class SlicedQuantity(Quantity):
    """ Quantity needing slots in an array """
    storage_name = None

    class Resource(Quantity.Resource):
        def __init__(self, *args, key=None, sub_slice=None, copy="", **kwargs):
            super().__init__(*args, **kwargs)
            self.key = key or self
            self.sub_slice = sub_slice
            self._copy = copy

        def expression(self, ode_builder, slc_prefix=""):
            slc = self.request_slice(ode_builder)
            if self.sub_slice is not None:
                if self.sub_slice == slice(None):
                    slc = structured_data.slicify(slc)
                else:
                    slc = structured_data.slc_compose(slc, self.sub_slice)
            return "{}[{}{}]{}".format(self.base.storage_name, slc_prefix, slc,
                                       self._copy)

        def register(self, ode_builder):
            self.request_slice(ode_builder)

        def request_slice(self, ode_builder):
            return ode_builder.request_slice(self.key, self.base.n, type(self.base))

        def __getitem__(self, slc):
            if self._copy:
                raise ValueError("A copied resource cannot be indexed")
            return type(self)(self.base, self.parent, self.alias, key=self.key, sub_slice=slc)


class StateFunction(SlicedQuantity):
    storage_name = "_x"
    array_name_dx = "_dx"

    class Resource(SlicedQuantity.Resource):
        def expression_dx(self, ode_builder, flatten_slice=True, slc_prefix=""):
            slc = self.request_slice(ode_builder)
            if not flatten_slice:
                slc = structured_data.slicify(slc)
            return "{}[{}{}]".format(self.base.array_name_dx, slc_prefix, slc)


class ManipulatedVariable(SlicedQuantity):
    storage_name = "_u"

    class Resource(SlicedQuantity.Resource):
        def copy(self):
            return type(self)(self.base, self.parent, self.alias, key=self.key,
                              sub_slice=self.sub_slice, copy=".copy()")


class MonitoredQuantity(SlicedQuantity):
    storage_name = "_y"


class ExternalQuantity(SlicedQuantity):
    """ External quanty, e.g. ambient temperature """
    storage_name = "_v"

    @classmethod
    def bound(cls, name, **kwargs):
        return cls(name, **kwargs).resource(None, None)

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self._resource = super().resource(None, name)

    def resource(self, *args, **kwargs):
        return self._resource


class MetaGlobalExternalQuantity(type):
    def __call__(cls, name, n=1, **kwargs):
        q = getattr(cls, name, None)
        if not isinstance(q, ExternalQuantity):
            q = ExternalQuantity(name=name, n=n)
        return q.resource(None, name)


class GlobalExternalQuantity(metaclass=MetaGlobalExternalQuantity):
    pass


class ForeignQuantity(Quantity):
    """ Quanty from a different element to be used in the owner element """

    def __init__(self):
        super().__init__(n=None)

    class Resource(Quantity.Resource):
        def __init__(self, *args, target=None, sub_slice=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.target = target
            self.sub_slice = sub_slice

        def slc_expression(self, slc_prefix=""):
            if self.sub_slice is None:
                return ""
            return "[{}{}]".format(slc_prefix, self.sub_slice)

        def register(self, ode_builder):
            self.target.register(ode_builder)

        def expression(self, ode_builder, slc_prefix=""):
            return self.target.expression(ode_builder, slc_prefix=slc_prefix) \
                + self.slc_expression(slc_prefix)

        def expression_dx(self, ode_builder, slc_prefix=""):
            return self.target.expression_dx(ode_builder, slc_prefix=slc_prefix) \
                + self.slc_expression(slc_prefix)

        def dependencies(self):
            return self.target.dependencies()

        def computations(self):
            return self.target.computations()

        def __repr__(self):
            return repr(self.target) + self.slc_expression()

    @classmethod
    def anonymous(cls, target=None):
        return cls().resource(None, None, target=target)


class ComputedQuantity(Quantity):
    def __init__(self, f_compute, **kwargs):
        sig = inspect.signature(f_compute)
        n = sig.return_annotation
        if not isinstance(n, int):
            raise ValueError(
                "The function has to be annotated with an int (dimension of the return value)")
        super().__init__(n=n)
        self.function = ComputedQuantityFunction(f_compute, **kwargs)

    def _resource(self, parent, *args, computation, **kwargs):
        rsc = super().resource(parent, *args, computation=computation, **kwargs)
        computation.return_resources = (rsc,)
        return rsc

    def resource(self, parent, *args, **kwargs):
        computation = self.function.computation(parent)
        return self._resource(parent, *args, computation=computation, **kwargs)

    def resource_by_args(self, parent, alias, **kwargs):
        computation = self.function.computation_by_args(**kwargs)
        return self._resource(parent, alias, computation=computation)

    class Resource(Quantity.Resource):
        def __init__(self, base, parent, alias, computation, **kwargs):
            super().__init__(base, parent, alias, **kwargs)
            expression = alias
            if parent.name:
                expression = "_".join(
                    tuple(parent.parent_names) + (parent.name, expression))
            self._expression = expression
            self.computation = computation

        def computations(self):
            return frozenset((self.computation,))

        def expression(self, ode_builder, slc_prefix=""):
            return self._expression

        def dependencies(self):
            return frozenset(self.computation.resource_args.values())


class Function:
    f_prefix = ""

    def __init__(self, f):
        sig = inspect.signature(f)
        self.check_signature(sig)
        self.f = f
        if sig.return_annotation is not inspect.Signature.empty:
            self.return_annotation = sig.return_annotation
        else:
            self.return_annotation = ()
        annotations = {arg: prm.annotation for arg,
                       prm in sig.parameters.items()
                       if prm.annotation is not inspect.Signature.empty}
        self.arguments = tuple(
            name for name, prm in sig.parameters.items() if name not in annotations)

    class Computation:
        call_template = "{}({})"
        operation_template = "{return_values} = {call}"

        def __init__(self, f, resource_args=None, return_resources=(), f_name=None,
                     block_args=()):
            self.f_name = f_name or f.__name__
            sig = inspect.signature(f)
            self.template = self.call_template.format("{__f__}", ", ".join(
                "{" + arg + "}" for arg in sig.parameters))
            self.call_template = self.template
            annotations = tuple((arg, d_.unwrap(prm.annotation)) for arg,
                                prm in sig.parameters.items()
                                if prm.annotation is not inspect.Signature.empty)
            tuple_annotations = ((arg, a) for arg, a in annotations
                                 if not isinstance(a, Quantity.Resource))
            invalid_annotations = tuple(arg for arg, a in tuple_annotations
                                        if not isinstance(a, tuple)
                                        or not all(isinstance(r, Quantity.Resource) for r in a))
            if invalid_annotations:
                raise ValueError(
                    "Invalid annotations (not of type Resource) for arguments: {}".format(
                        ", ".join(invalid_annotations)))
            _resource_args = dict(annotations)
            if resource_args is not None:
                assert set(resource_args).isdisjoint(set(_resource_args))
                _resource_args.update(resource_args)
            return_annotation = sig.return_annotation

            if block_args is not None:
                for arg in block_args:
                    _resource_args[arg] = _resource_args[arg].copy()
            if return_annotation is not inspect.Signature.empty:
                if not isinstance(return_annotation, tuple):
                    return_annotation = (return_annotation,)
                n_ret = len(return_annotation)
                self.n_ret = n_ret
                self.template = self.operation_template.format(
                    return_values=", ".join(("{}",) * n_ret), call=self.template)
                return_annotation = tuple(d_.unwrap(a)
                                          for a in return_annotation)
                if any(isinstance(r, Quantity.Resource) for r in return_annotation):
                    return_resources = tuple((r if isinstance(r, Quantity.Resource)
                                              else return_resources[i])
                                             for i, r in enumerate(return_annotation))
            else:
                self.n_ret = 0
            self.resource_args = _resource_args
            self.return_resources = return_resources
            self.f = f

        def computation_dependencies(self):
            return collect_computations(self.dependencies())

        def dependencies(self):
            deps = set()
            for r in self.resource_args.values():
                if isinstance(r, tuple):
                    deps |= set(r)
                else:
                    deps.add(r)
            return deps

        def args_to_resources(self, ode_builder, slc_prefix=""):
            def _expression(r, ode_builder):
                if isinstance(r, tuple):
                    return ", ".join(_r.expression(ode_builder, slc_prefix=slc_prefix) for _r in r)
                return r.expression(ode_builder, slc_prefix=slc_prefix)
            return {arg: _expression(r, ode_builder)
                    for arg, r in self.resource_args.items()}

        def return_values(self, ode_builder):
            return tuple(ret.expression(ode_builder)
                         for ret in self.return_resources)

        def write_code_by_args(self, env, returns=(), **subst_args):
            collisions = set(env) & set(returns)
            if collisions:
                raise ValueError(
                    "Name collision: trying to assign to '{}' which is already defined".format(
                        ", ".join(collisions)))
            f_name = self.f_name
            update_env(env, f_name, self.f)
            return self.template.format(*returns, __f__=f_name, **subst_args)

        def write_code(self, ode_builder, env, slc_prefix=""):
            subst_args = self.args_to_resources(
                ode_builder, slc_prefix=slc_prefix)
            returns = self.return_values(ode_builder)
            return self.write_code_by_args(env, returns=returns, **subst_args)

    def computation_by_args(self, resource_args=None, **kwargs):
        resource_args = resource_args or {}
        _resource_args = set(resource_args)
        missing_args = set(self.arguments) - _resource_args
        if missing_args:
            raise ValueError(
                "Missing quantity arguments for arguments {}".format(", ".join(missing_args)))
        unexpected_args = _resource_args - set(self.arguments)
        if unexpected_args:
            raise ValueError(
                "Unexpected quantity arguments {}".format(", ".join(unexpected_args)))
        return self.Computation(self.f, resource_args=resource_args, **kwargs)

    def fetch_args(self, parent):
        return {arg: (get_resource_recursive(parent, arg)
                      if arg != "self" else parent)
                for arg in self.arguments}

    def computation(self, parent, **kwargs):
        resource_args = self.fetch_args(parent)
        f_name = self.f.__name__
        if parent.name:
            f_name = "_".join(
                tuple(parent.parent_names) + (parent.name, f_name))
        f_name = self.f_prefix + f_name
        return_resources = tuple(get_resource_recursive(parent, val)
                                 for val in self.return_names()
                                 if not isinstance(val, Quantity.Resource))
        return self.computation_by_args(resource_args=resource_args,
                                        return_resources=return_resources,
                                        f_name=f_name,
                                        **kwargs)

    def return_names(self):
        return_annotation = self.return_annotation or ()
        if not isinstance(return_annotation, tuple):
            return (return_annotation,)
        return return_annotation

    def check_signature(self, sig):
        pass


def get_resource_recursive(e, addr):
    *addr, q = addr.split(".")
    target = e
    for part in addr:
        target = getattr(target, part)
    return target.__resources__[q]


class ComputedQuantityFunction(Function):
    f_prefix = "f_"

    def return_names(self):
        return ()


class Derivative(Function):
    def __init__(self, f, **kwargs):
        super().__init__(f, **kwargs)
        if self.return_annotation is inspect.Signature.empty:
            raise ValueError(
                "DxCallbacks must have a return annotation indicating"
                " the quantity / quantities beeing changed")
        if not (isinstance(self.return_annotation, tuple)
                and all(isinstance(a, d_) for a in self.return_annotation)) \
                and not isinstance(self.return_annotation, d_):
            raise ValueError(
                "The return annotation has to be a d_ instance")

    def return_names(self):
        return tuple(d_.unwrap(ret) for ret in super().return_names())

    class Computation(Function.Computation):
        operation_template = "{return_values} += {call}"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.n_ret > 1:
                temp_vars = tuple("_tmp" + str(i) for i in range(self.n_ret))
                assignments = ("{} += " + tmp for tmp in temp_vars)
                self.template = "{} = {call} ; {}".format(
                    ", ".join(temp_vars), " ; ".join(assignments), call=self.call_template)

        def return_values(self, ode_builder):
            return tuple(ret.expression_dx(ode_builder)
                         for ret in self.return_resources)

        def dependencies(self):
            return super().dependencies() | set(self.return_resources)


class DerivativeInline(Function):

    class Computation(Function.Computation):
        def __init__(self, f, *args, **kwargs):
            super().__init__(f, *args, **kwargs)
            self.diff_args = frozenset(d_.get_args(f))
            for x in self.diff_args:
                if not isinstance(self.resource_args[x], StateFunction.Resource):
                    raise TypeError(
                        "{} is marked as d_ quantity"
                        " but is no StateFunction".format(x))

        def args_to_resources(self, ode_builder, slc_prefix=""):
            resource_args = self.resource_args
            subst_args = {arg: r.expression(ode_builder, slc_prefix=slc_prefix)
                          for arg, r in self.resource_args.items() if arg not in self.diff_args}
            for arg in self.diff_args:
                subst_args[arg] = resource_args[arg].expression_dx(
                    ode_builder, flatten_slice=False, slc_prefix=slc_prefix)
            return subst_args


class d_:
    def __init__(self, addr):
        self.addr = addr

    @classmethod
    def unwrap(cls, x):
        if isinstance(x, cls):
            return x.addr
        return x

    @classmethod
    def get_args(cls, f):
        sig = inspect.signature(f)
        return (arg for arg, prm in sig.parameters.items()
                if isinstance(prm.annotation, cls))


class CtrlFunction(Function):
    pass


class ODEBuilder:
    def __init__(self, rsc_t=None, **kwargs):
        self._slice_factories = {}
        if rsc_t is not None:
            rsc_t.register(self)
        for r in collect_resources_(**kwargs):
            r.register(self)

    @classmethod
    def setup_ode(cls, t, derivative_computations,
                  env=None, ctrl_callback=None, ode_builder=None,
                  f_ctrl_get=None,
                  src_callback=None,
                  **kwargs):
        dx_env = {}
        ode_builder = ode_builder or cls()
        computations = collect_computations(collect_resources_(
            derivative_computations, ctrl_callback=ctrl_callback))
        dependencies = {computation: computation.computation_dependencies()
                        for computation in computations}
        computation_jobs = depcalc.depcalc(dependencies)
        code_lines = ode_builder.write_ode_code_by_args(
            computation_jobs, derivative_computations, dx_env=dx_env, ctrl_callback=ctrl_callback,
            f_ctrl_get=f_ctrl_get,
            src_callback=src_callback,
            **kwargs)
        ode_builder.allocate_resources(dx_env)
        if env is not None:
            dx_env.update(env)
        # print(code_generator.line_numbered_code_lines(code_lines))
        _dx = code_generator.Function.build(name="d_X", args=(StateFunction.storage_name, "t"),
                                            code_lines=code_lines,
                                            environment=dx_env)
        _dx.named_lengths = dict(ode_builder.named_lengths())
        _dx.quantity_lengths = ode_builder.quantity_lengths()
        _dx.src = src_callback
        _dx.ctrl = f_ctrl_get
        _dx.t = t
        _dx.code_lines = code_lines
        _dx.u = dx_env.get(ManipulatedVariable.storage_name)
        return _dx

    def write_ode_code_by_args(self, computation_jobs, derivative_computations,
                               dx_env,
                               f_ctrl_get=None,
                               ctrl_callback=None,
                               src_callback=None):
        dx_code = tuple(d.write_code(self, dx_env)
                        for d in derivative_computations)

        computation_code = tuple(function.write_code(self, dx_env)
                                 for function in computation_jobs)
        ctrl_code = self.write_ctrl_code(dx_env, ctrl_callback)
        code_lines = self.write_src_code(dx_env, src_callback) \
            + self.write_ctrl_get_code(dx_env, f_ctrl_get) \
            + computation_code \
            + ctrl_code \
            + (StateFunction.array_name_dx + "[:] = 0.",) \
            + dx_code \
            + ("return " + StateFunction.array_name_dx,)
        return code_lines

    @classmethod
    def monitor_call(cls, computed_resources,
                     env=None, ode_builder=None, **kwargs):
        ode_builder = ode_builder or cls()
        _env = {}
        named_lengths = tuple((r, r.base.n) for r in computed_resources)
        out_slices = tuple(structured_data.xtuple_len_to_slc(
            named_lengths, flatten_slice=True))
        expr_slices = ((r.expression(ode_builder), slc)
                       for r, slc in out_slices)
        computations = collect_computations(
            collect_resources(computed_resources))
        dependencies = {computation: computation.computation_dependencies()
                        for computation in computations}
        computation_jobs = depcalc.depcalc(dependencies)
        code_lines = ode_builder.write_monitor_code_by_args(
            computation_jobs, env=_env, out_slices=expr_slices)
        # print(code_generator.line_numbered_code_lines(code_lines))
        if env is not None:
            _env.update(env)
        _Y = code_generator.Function.build(name="Y", args=("t", MonitoredQuantity.storage_name,
                                                           StateFunction.storage_name,
                                                           ManipulatedVariable.storage_name,
                                                           ExternalQuantity.storage_name),
                                           code_lines=code_lines,
                                           environment=_env)
        _Y.named_lengths = tuple((repr(r), n) for r, n in named_lengths)
        return _Y

    def write_monitor_code_by_args(self, computation_jobs, out_slices, env):
        code_lines = tuple(function.write_code(self, env, slc_prefix="..., ")
                           for function in computation_jobs) \
            + tuple("{}[..., {}] = {}.squeeze()".format(MonitoredQuantity.storage_name, slc, expr)
                    for expr, slc in out_slices)
        return code_lines

    def named_lengths(self, quantity_type=None):
        if quantity_type is None:
            return dict((storage_type, factory.named_lengths_repr())
                        for storage_type, factory in self._slice_factories.items())
        return self._slice_factories[quantity_type].named_lengths_repr()

    def quantity_lengths(self, quantity_type=None):
        if quantity_type is None:
            return {q_type: fct.named_lengths for q_type, fct in self._slice_factories.items()}
        return self._slice_factories[quantity_type].named_lengths

    def request_slice(self, key, n, storage_type):
        factory = self.lookup_slice_factory(storage_type)
        return factory.add_named_length(key, n)

    def lookup_slice_factory(self, storage_type):
        factory = self._slice_factories.get(storage_type)
        if factory is not None:
            return factory
        factory = SliceFactory()
        self._slice_factories[storage_type] = factory
        return factory

    def allocate_resources(self, env):
        L = self._slice_factories[StateFunction].total_length
        update_env(env, StateFunction.array_name_dx, numpy.empty(L))
        slc_fac = self._slice_factories.get(ManipulatedVariable)
        if slc_fac is not None:
            L = slc_fac.total_length
            update_env(env, ManipulatedVariable.storage_name,
                       numpy.empty(L))

    def write_src_code(self, env, src_callback=None):
        if src_callback is None:
            return ()
        storage_type = ExternalQuantity
        v_storage = storage_type.storage_name
        slice_info = self._slice_factories.get(storage_type)
        if slice_info is None:
            return ()
        update_env(env, v_storage, numpy.empty(slice_info.total_length))
        update_env(env, "_src", src_callback)
        return ("_src(t, {})".format(v_storage),)

    @staticmethod
    def write_ctrl_get_code(env, f_ctrl_get=None):
        u_storage = ManipulatedVariable.storage_name
        if f_ctrl_get is None:
            return ()
        update_env(env, "_ctrl_get", f_ctrl_get)
        return ("{}, trigger_ctrl = _ctrl_get(t)".format(u_storage),)

    def write_ctrl_code(self, env, ctrl_callback=None):
        if ctrl_callback is None:
            return ()
        return ("if trigger_ctrl: " + ctrl_callback.write_code(self, env),)


def update_env(env, key, val):
    if key in env:
        raise ValueError("Name collision for {}".format(key))
    env[key] = val


def collect_computations(resources):
    return resources and frozenset.union(*(r.computations() for r in resources))


def collect_resources_from_functions(functions):
    if not functions:
        return set()
    args = set.union(*(set(f.dependencies()) for f in functions))
    return collect_resources(args)


def collect_resources_(functions=None, ctrl_callback=None, computed_resources=None):
    resources = set()
    if functions:
        resources |= collect_resources_from_functions(
            functions)
    if ctrl_callback is not None:
        resources |= collect_resources_from_functions(
            (ctrl_callback,))
    if computed_resources:
        resources |= collect_resources(computed_resources)
    return resources


def collect_resources(resources):
    remaining_args = resources.copy()
    args = resources.copy()
    while remaining_args:
        arg = remaining_args.pop()
        new_args = arg.dependencies()
        new_args -= args
        remaining_args |= new_args
        args |= new_args
    return args


class SliceFactory:
    def __init__(self):
        self.slices = {}
        self.total_length = 0
        self.named_lengths = []
        self._factory = structured_data.slc_enumerator(
            slice_type=structured_data.flat_slice_l)

    def add_named_length(self, key, length):
        slc = self.slices.get(key)
        if slc is not None:
            return slc
        self.named_lengths.append((key, length))
        self.total_length += length
        slc = self._factory(length)
        self.slices[key] = slc
        return slc

    def named_lengths_repr(self):
        return tuple((repr(r), l) for r, l in self.named_lengths)


def collect(content, mapping):
    for type_, dest in mapping.items():
        dest.update({key: val for key, val in content
                     if isinstance(val, type_)})


class QuantityContainerMeta(type):
    def __new__(mcs, name, bases, classdict):
        quantities = {}
        derivatives = {}
        for base in bases:
            quantities.update(base.__quantities__)
            derivatives.update(base.__derivatives__)
        mapping = {Quantity: quantities, Derivative: derivatives,
                   DerivativeInline: derivatives}
        collect(classdict.items(), mapping)
        classdict["__derivatives__"] = derivatives
        classdict["__quantities__"] = quantities

        return type.__new__(mcs, name, bases, classdict)

    def __call__(cls, *args, **kwargs):
        instance = type.__call__(cls, *args, **kwargs)
        return instance


class QuantityContainer(metaclass=QuantityContainerMeta):

    def __new__(cls, *args, name=None, parent_names=(), **kwargs):
        instance = super().__new__(cls)
        quantities = cls.__quantities__.copy()
        instance.__quantities__ = quantities
        instance.__derivatives__ = cls.__derivatives__.copy()
        instance.__resources__ = {}
        instance.__derivative_computations__ = {}
        instance.__sub_elements__ = {}
        instance.name = name
        instance.parent_names = parent_names
        dependencies = {(n, q): set((arg, quantities[arg])
                                    for arg in q.function.arguments
                                    if arg in quantities)
                        for n, q in quantities.items()
                        if isinstance(q, ComputedQuantity)}
        for name, q in depcalc.deporder(tuple(quantities.items()), dependencies):
            setattr(instance, name, q.resource(parent=instance, alias=name))
        for name, d in instance.__derivatives__.items():
            computation = d.computation(instance)
            instance.__derivative_computations__[name] = computation
            setattr(instance, name, computation)
        return instance

    def __init__(self, name=None, parent_names=()):
        pass

    def __repr__(self):
        _repr = self.name or super().__repr__()
        return ".".join(tuple(self.parent_names) + (_repr,))

    def __setattr_quantity__(self, key, val):
        self.__quantities__[key] = val
        return self.__setattr_resource__(key, val.resource(parent=self, alias=key))

    def __setattr_resource__(self, key, val):
        self.__resources__[key] = val
        return val

    def __setattr_derivative__(self, key, val):
        self.__derivatives__[key] = val
        return self.__setattr_computation__(key, val.computation(self))

    def __setattr_computation__(self, key, val):
        self.__derivative_computations__[key] = val
        return val

    def __setattr_element__(self, key, val):
        self.__sub_elements__[key] = val
        return val

    def __setattr__(self, key, val):
        val = self.set_child(key, val)
        super().__setattr__(key, val)

    def set_child(self, key, val):
        typemapping = {Quantity: self.__setattr_quantity__,
                       Quantity.Resource: self.__setattr_resource__,
                       Derivative: self.__setattr_derivative__,
                       DerivativeInline: self.__setattr_derivative__,
                       Derivative.Computation: self.__setattr_computation__,
                       DerivativeInline.Computation: self.__setattr_computation__,
                       Element: self.__setattr_element__}
        for type_, _setatr in typemapping.items():
            if isinstance(val, type_):
                return _setatr(key, val)
        return val

    def discard_child(self, key):
        if key in self.__resources__:
            del self.__resources__[key]
        if key in self.__derivatives__:
            del self.__derivatives__[key]
        if key in self.__sub_elements__:
            del self.__sub_elements__[key]

    def __delattr__(self, key):
        super().__delattr__(key)
        self.discard_child(key)


class Model(QuantityContainer):

    t = ExternalQuantity(name="t", n=1)

    def __init__(self, **kwargs):
        self.add(**kwargs)

    def add(self, **kwargs):
        for key, arg in kwargs.items():
            setattr(self, key, arg)

    @classmethod
    def from_file(cls, file_name, *args, **kwargs):
        with open(file_name) as f:
            return cls.from_stream(f, *args, **kwargs)

    @classmethod
    def from_stream(cls, stream, namespace):
        return model_json.model_from_stream(stream, namespace, model_class=cls,
                                            inject_name=frozenset(
                                                (Element,
                                                 GlobalExternalQuantity,
                                                 ConstantQuantity)),
                                            inject_parent_names=frozenset(
                                                (Element,)),
                                            foreign_quantity_wrapper=ForeignQuantity.anonymous)

    @staticmethod
    def resource(type_):
        def _resources(e):
            return (resource for resource in e.__resources__.values()
                    if isinstance(resource.base, type_))
        return _resources

    @staticmethod
    def _derivatives(e):
        return e.__derivative_computations__.values()

    def derivatives(self):
        return self.collect_element_resources(
            self, self._derivatives)

    def computed_resources(self):
        return self.collect_element_resources(
            self, self.resource(ComputedQuantity))

    @classmethod
    def collect_element_resources(cls, e, type_, out=None):
        if out is None:
            out = set()
        out |= set(type_(e))
        for child in e.__sub_elements__.values():
            cls.collect_element_resources(child, type_, out=out)
        return out

    def external_resources(self):
        return self.collect_element_resources(
            self, self.resource(ExternalQuantity))

    def setup_ode(self, f_ctrl=None, block_ctrl_args=None, **kwargs):
        ctrl_callback = f_ctrl and CtrlFunction(f_ctrl).computation(self,
                                                                    block_args=block_ctrl_args)
        derivative_computations = self.derivatives()
        return ODEBuilder.setup_ode(self.t, derivative_computations,
                                    ctrl_callback=ctrl_callback,
                                    **kwargs)

    def monitor_call(self, monitored_arguments=None, **kwargs):
        if monitored_arguments is None:
            monitored_resources = self.computed_resources()
        else:
            monitored_resources = tuple(
                getattr_recursive(self, arg) for arg in monitored_arguments)
            wrong_args = tuple(arg for arg, r in zip(monitored_arguments, monitored_resources)
                               if not isinstance(r, ComputedQuantity.Resource))
            if wrong_args:
                raise ValueError("Arguments {} are not computed resources".format(
                    ", ".join(".".join(arg) for arg in wrong_args)))
        return ODEBuilder.monitor_call(monitored_resources, **kwargs)

    def compile(self, f_ctrl=None):
        sources = timeseries_interpolant()
        monitor_call_args = {"src_callback": sources}
        ode_args = {"src_callback": sources}
        if f_ctrl is not None:
            ctrl_resources = collect_resources_from_functions((
                CtrlFunction(f_ctrl).computation_by_args(),))
            ctrl = controller()
            monitor_call_args["f_ctrl_get"] = ctrl
            ode_args["f_ctrl_get"] = ctrl
            ode_args["f_ctrl"] = f_ctrl
        else:
            ctrl_resources = set()
        ode_builder = ODEBuilder(
            rsc_t=self.t,
            functions=self.derivatives(),
            computed_resources=ctrl_resources | self.computed_resources())
        d_X = self.setup_ode(
            ode_builder=ode_builder, **ode_args)
        Y = self.monitor_call(
            ode_builder=ode_builder, **monitor_call_args)
        return d_X, Y

    def run(self, initial_data, t, f_ctrl=None, **kwargs):
        d_X, Y = self.compile(f_ctrl=f_ctrl)

        return simulate(initial_data, t, d_X, Y=Y, **kwargs)


def solve_ode(d_X, x0, t, tcrit=None, Y=None, exception_on_error=False, out=None):
    Nx = x0.shape[0]
    if out is None:
        if Y is not None:
            Ny = sum(l for n, l in Y.named_lengths)
        else:
            Ny = 0
        out = numpy.empty((t.size, Nx + Ny))
    _x = scipy.integrate.odeint(d_X, x0, t, tcrit=tcrit,
                                atol=1.49012e-15,
                                rtol=1.49012e-6,
                                full_output=exception_on_error)
    if exception_on_error:
        res = _x[1]
        _x = _x[0]
        if res['message'] != 'Integration successful.':
            raise RuntimeError('scipy.odeint failed.')
    out[:, :Nx] = _x
    if Y is not None:
        _v = numerics.interp_linear(
            d_X.src.t, d_X.src.dat, t)
        u = d_X.ctrl and numerics.interp_step(d_X.ctrl.t, d_X.ctrl.u, t)
        if u is None:
            u = d_X.u
        Y(t, out[:, Nx:], _x, u, _v)
    return out


def simulate(initial_data, t, d_X, Y=None, tcrit=None, t_ctrl_sampler=None,
             external_quantity_data=None, **kwargs):
    x0 = arrange_data(initial_data, d_X.quantity_lengths[StateFunction])
    u_lengths = d_X.quantity_lengths.get(ManipulatedVariable)
    if u_lengths is not None:
        u0 = arrange_data(
            initial_data, u_lengths)
        d_X.u[()] = u0
        if d_X.ctrl is not None:
            if t_ctrl_sampler is not None:
                t_ctrl = t_ctrl_sampler(t)
                d_X.ctrl.set_data(t_ctrl, u0)
            else:
                t_ctrl = d_X.ctrl.t
            if tcrit is not None:
                tcrit = numpy.unique(numpy.concatenate((t_ctrl, tcrit)))
            else:
                tcrit = t_ctrl
    if external_quantity_data is not None:
        t_src = external_quantity_data[d_X.t]
        data = arrange_data(
            external_quantity_data, d_X.quantity_lengths[ExternalQuantity],
            extra_shape=(t_src.size,))
        d_X.src.set_data(t_src, data)
    x = solve_ode(d_X, x0, t, Y=Y, tcrit=tcrit, **kwargs)

    if len(t) != x.shape[0]:
        u = None
        if d_X.ctrl is not None:
            u = d_X.ctrl.u[d_X.ctrl.end - 1]
        warnings.warn(
            "Sizes of t and x do not agree. odeint probably failed. Dump:\n{}".format(
                dump(t[x.shape[0]], x[:, -1], u, d_X)))

    def compose_timeseries(named_lengths, t, data):
        named_views = structured_data.xtuple_len_to_view(
            named_lengths, data, flatten_slice=True)
        return [(name, (t, view)) for name, view in named_views]
    Y_series = ()
    if Y is not None:
        Y_series = Y.named_lengths
    time_series = compose_timeseries(
        d_X.named_lengths[StateFunction] + Y_series,
        t[:x.shape[0]], x) \
        + compose_timeseries(d_X.named_lengths[ExternalQuantity][1:], d_X.src.t,
                             d_X.src.dat[:, 1:])
    if d_X.ctrl is not None:
        ctrl = d_X.ctrl
        end = ctrl.end
        t_u = ctrl.t[:end]
        u = ctrl.u[:end]
        time_series += compose_timeseries(
            d_X.named_lengths[ManipulatedVariable], t_u, u)
    return dct.dict_assert_unique(time_series, dict_class=OrderedDict)


def arrange_data(data, dest_named_lengths, extra_shape=()):
    dest_slices = tuple(structured_data.xtuple_len_to_slc(dest_named_lengths))
    x0 = numpy.empty(
        extra_shape + (structured_data.slc_stop(dest_slices[-1][1]),))
    for f, slc in dest_slices:
        try:
            x0[..., slc] = numpy.array(
                data[f], copy=False).reshape(x0[..., slc].shape)
        except KeyError:
            raise KeyError('"{}" not found in initial data'.format(f))
        except ValueError:
            raise ValueError(
                'Columnn width missmatch for column "{}". Expected {}, got {}'.format(
                    f,
                    x0[..., slc].shape,
                    data[f].shape))
    return x0


def dump(t, x, u, d_X):
    return "t = {}\n{}\n{}\n{}\n{}\n".format(t,
                                             structured_data.csv_header(
                                                 d_X.named_lengths[StateFunction]),
                                             ';'.join(map(str, x)),
                                             structured_data.csv_header(
                                                 d_X.named_lengths[ManipulatedVariable]),
                                             ';'.join(map(str, u)))


class Element(QuantityContainer):
    def __init__(self, name, parent_names=()):
        super().__init__(name=name, parent_names=parent_names)

    def __new__(cls, name, *args, **kwargs):
        return super().__new__(cls, *args, name=name, **kwargs)


def timeseries_interpolant(*args):
    """
    Returns an interpolant for the timeseries dat with timestamps t
    """
    _i_current = 1
    _n = None
    _d_dat = ()
    _dat = ()
    _t = ()

    def _interp(t, out=None):
        """
        Interpolate all quantities in a record for time t
        (extrapolate it if t is past the last time in the records)
        """
        nonlocal _i_current, _n
        while t > _t[_i_current] and _i_current < _n - 1:
            _i_current += 1
        while t < _t[_i_current - 1] and _i_current > 1:
            _i_current -= 1
        i = _i_current - 1
        dt = t - _t[i]
        if out is None:
            out = numpy.empty(_dat.shape[-1])
        out[:] = _d_dat[i]
        out *= dt
        out += _dat[i]
        return out

    def _interp_vector(t):
        return numerics.interp_linear(_t, _dat, t)

    def set_data(t, dat):
        nonlocal _t, _i_current, _n, _d_dat, _dat
        _i_current = 1
        _n = dat.shape[0]
        _d_dat = (dat[1:] - dat[:-1]) / (t[1:, None] - t[:-1, None])
        _t = t
        _dat = dat
        _interp.dat = dat
        _interp.t = t

    if args:
        set_data(*args)
    _interp.set_data = set_data
    _interp.interpolate = _interp_vector
    return _interp


def t_sampler(Delta_t):
    def t_samples(t):
        return numpy.arange(t[0], t[-1], Delta_t)
    return t_samples


def controller(*args):

    max_end = None
    _end = None
    _t = ()
    _u = ()
    _i = None

    def _get(t):
        """ Step interpolate in values already set """
        nonlocal _i, _end
        while _i < max_end - 1 and t >= _t[_i + 1]:
            _i += 1
        while t < _t[_i] and _i > 0:
            _i -= 1
        trigger_ctrl = (_i >= _end)
        if trigger_ctrl:
            if _i != _end:
                warnings.warn("Controller: missed {} controlling cycles".format(
                              _i - _end))
            _u[_end:_i + 1] = _u[_end - 1]
            _end = _i + 1
            _get.end = _end

        return _u[_i], trigger_ctrl

    def set_data(t, u0):
        nonlocal max_end, _u, _t, _i, _end
        n = u0.shape[0]
        max_end = t.shape[0]
        _t = t
        _u = numpy.empty((max_end, n))
        _u[0] = u0
        _i = 0
        _end = 1
        _get.end = _end
        _get.t = _t
        _get.u = _u

    if args:
        set_data(*args)

    _get.set_data = set_data
    return _get
