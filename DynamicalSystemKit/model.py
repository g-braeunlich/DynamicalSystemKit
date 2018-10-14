import inspect
import warnings
from collections import OrderedDict
import numpy
import scipy

from . import code_generator, numerics
from .utils import structured_data, depcalc, dct
from .utils.attr import getattr_recursive


class Node:
    def __init__(self, parent=None, alias=None):
        self.parent = parent
        self.alias = alias

    def __repr__(self):
        parent_repr = self.parent and (repr(self.parent), )
        alias = self.alias or object.__repr__(self)
        return ".".join(((parent_repr or ()) + (alias,)))

    def bind_to_parent(self, parent, alias):
        self.parent = parent
        self.alias = alias


class Expressible:
    def expression(self, ode_builder, slc_prefix=""):
        pass

    def expression_view(self, ode_builder, slc_prefix=""):
        return self.expression(ode_builder, slc_prefix=slc_prefix)


class Quantity(Node, Expressible):
    """ Base class for quantities (state function, control quantity) """

    def __init__(self, n=1, **kwargs):
        self.n = n  #: Dimension of the quantities vector space
        super().__init__(**kwargs)

    def register(self, ode_builder):
        pass

    def dependencies(self):
        return frozenset()

    def computations(self):
        return frozenset()


class ConstantQuantity(Quantity):
    def __init__(self, value):
        self.value = value
        shp = numpy.shape(value)
        if shp:
            [n] = shp
        else:
            n = 1
        super().__init__(n=n)

    def expression(self, ode_builder, slc_prefix=""):
        return str(self.value)


class SlicedQuantity(Quantity):
    """ Quantity needing slots in an array """
    storage_name = None

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
        return "{}[{}{}]{}".format(self.storage_name, slc_prefix, slc,
                                   self._copy)

    def register(self, ode_builder):
        self.request_slice(ode_builder)

    def request_slice(self, ode_builder):
        return ode_builder.request_slice(self.key, self.n, type(self))

    def __getitem__(self, slc):
        if self._copy:
            raise ValueError("A copied quantity cannot be indexed")
        return type(self)(n=self.n, parent=self.parent, alias=self.alias,
                          key=self.key,
                          sub_slice=slc)


class StateFunction(SlicedQuantity):
    storage_name = "_x"
    array_name_dx = "_dx"

    def expression_dx(self, ode_builder, flatten_slice=True, slc_prefix=""):
        slc = self.request_slice(ode_builder)
        if not flatten_slice:
            slc = structured_data.slicify(slc)
        return "{}[{}{}]".format(self.array_name_dx, slc_prefix, slc)


class ManipulatedVariable(SlicedQuantity):
    storage_name = "_u"

    def copy(self):
        return type(self)(n=self.n, parent=self.parent, alias=self.alias, key=self.key,
                          sub_slice=self.sub_slice, copy=".copy()")


class MonitoredQuantity(SlicedQuantity):
    storage_name = "_y"


class ExternalQuantity(SlicedQuantity):
    """ External quanty, e.g. ambient temperature """
    storage_name = "_v"


class ComputedQuantity(Quantity):
    def __init__(self, f_compute, **kwargs):
        sig = inspect.signature(f_compute)
        n = sig.return_annotation
        if not isinstance(n, int):
            raise ValueError(
                "The function has to be annotated with an int (dimension of the return value)")
        super().__init__(n=n, **kwargs)
        self.function = ComputedQuantityFunction(
            f_compute, return_quantities=(self,), **kwargs)
        self._f = f_compute

    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)

    def computations(self):
        return frozenset((self.function,))

    def expression(self, ode_builder, slc_prefix=""):
        if self.parent and self.parent.name:
            return "_".join(
                tuple(self.parent.parent_names) + (self.parent.name, self.alias))
        return self.alias

    def dependencies(self):
        return self.function.dependencies()


class Function(Node):
    f_prefix = ""
    operation = "="

    def __init__(self, f, *args, return_quantities=None, **kwargs):
        super().__init__(*args, **kwargs)
        sig = inspect.signature(f)
        self.f = f
        annotations = tuple((arg, prm.annotation) for arg,
                            prm in sig.parameters.items())
        missing_annotations = tuple(arg for arg, a in annotations
                                    if a is inspect.Signature.empty)
        if missing_annotations:
            raise ValueError(
                "Missing annotations for arguments: {}".format(
                    ", ".join(missing_annotations)))
        invalid_annotations = tuple(arg for arg, a in annotations
                                    if not isinstance(a, Expressible)
                                    and not (isinstance(a, tuple)
                                             and all(isinstance(q, Expressible) for q in a)))
        if invalid_annotations:
            raise ValueError(
                "Invalid annotations (not of type Expressible) for arguments: {}".format(
                    ", ".join(invalid_annotations)))
        self.arguments = sum((tuplify(a) for _, a in annotations), ())

        if return_quantities is None:
            return_quantities = sig.return_annotation
            if return_quantities is inspect.Signature.empty:
                return_quantities = ()
            elif not isinstance(return_quantities, tuple):
                return_quantities = (return_quantities,)
        self.check_return_quantities(return_quantities)
        self.return_quantities = return_quantities
        # TODO: blockargs
        # if block_args is not None:
        #     for arg in block_args:
        #         _resource_args[arg] = _resource_args[arg].copy()

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    @classmethod
    def check_return_quantities(cls, return_objects):
        invalid_annotations = tuple(
            str(i) for i, a in enumerate(return_objects)
            if not isinstance(a, Quantity))
        if invalid_annotations:
            raise ValueError(
                "Invalid return annotation. Objects at positions {}"
                " are not of type Quantity".format(
                    ", ".join(invalid_annotations)))

    def computation_dependencies(self):
        return collect_computations(self.dependencies())

    def dependencies(self):
        return set(d_.unwrap(a) for a in self.arguments)

    def write_code(self, ode_builder, env, slc_prefix=""):
        returns = self.return_expressions(ode_builder)
        return self._write_code(ode_builder, env, slc_prefix=slc_prefix,
                                return_expressions=returns, operation=self.operation)

    def _write_code(self, ode_builder, env, *, slc_prefix="", return_expressions, operation):
        collisions = set(env) & set(return_expressions)
        if collisions:
            raise ValueError(
                "Name collision: trying to assign to '{}' which is already defined".format(
                    ", ".join(collisions)))
        f_name = self.f_name()
        update_env(env, f_name, self.f)
        code = f_name + \
            "(" + ", ".join(a.expression_view(ode_builder, slc_prefix=slc_prefix)
                            for a in self.arguments) + ")"
        if return_expressions:
            code = ", ".join(return_expressions) + " " + \
                operation + " " + code
        return code

    def return_expressions(self, ode_builder):
        return tuple(ret.expression(ode_builder)
                     for ret in self.return_quantities)

    def f_name(self):
        _name = self.f.__name__
        if self.parent is not None and self.parent.name:
            _name = "_".join(
                tuple(self.parent.parent_names) + (self.parent.name, _name))
        _name = self.f_prefix + _name
        return _name


class ComputedQuantityFunction(Function):
    f_prefix = "f_"


class DerivativeBase(Function):
    pass


class Derivative(DerivativeBase):
    operation = "+="

    def write_code(self, ode_builder, env, slc_prefix=""):
        n_ret = len(self.return_quantities)
        if n_ret > 1:
            temp_vars = tuple("_tmp" + str(i) for i in range(n_ret))
            code = self._write_code(ode_builder, env, slc_prefix=slc_prefix,
                                    return_expressions=temp_vars, operation="=")
            assignments = (ret + " += " + tmp for ret,
                           tmp in zip(self.return_expressions(ode_builder), temp_vars))
            code += " ; {}".format(" ; ".join(assignments))
            return code
        return super().write_code(ode_builder, env, slc_prefix=slc_prefix)

    @classmethod
    def check_return_quantities(cls, return_objects):
        if not return_objects:
            raise ValueError(
                "Derivatives must have a return annotation indicating"
                " the quantity / quantities beeing changed")
        if not (isinstance(return_objects, tuple)
                and all(isinstance(a, d_) for a in return_objects)) \
                and not isinstance(return_objects, d_):
            raise ValueError(
                "The return annotation has to be a d_ instance")
        invalid_annotations = tuple(
            str(i) for i, a in enumerate(return_objects)
            if not isinstance(a, d_))
        if invalid_annotations:
            raise ValueError(
                "Invalid return annotation. Objects at positions {}"
                " are not of type d_".format(
                    ", ".join(invalid_annotations)))

    def dependencies(self):
        return super().dependencies() | frozenset(d_.unwrap(r) for r in self.return_quantities)


class DerivativeInline(DerivativeBase):

    @classmethod
    def check_return_quantities(cls, return_objects):
        if return_objects:
            raise ValueError(
                "DerivativeInline does not accept return annotations")


class d_(Expressible):
    def __init__(self, quantity):
        if not isinstance(quantity, StateFunction):
            raise TypeError(
                "Only StateFunction can be differentiated")
        self.quantity = quantity

    def expression(self, ode_builder, slc_prefix=""):
        return self.quantity.expression_dx(ode_builder, slc_prefix=slc_prefix,
                                           flatten_slice=True)

    def expression_view(self, ode_builder, slc_prefix=""):
        return self.quantity.expression_dx(ode_builder, slc_prefix=slc_prefix,
                                           flatten_slice=False)

    @classmethod
    def unwrap(cls, x):
        if isinstance(x, cls):
            return x.quantity
        return x


class CtrlFunction(Function):
    pass


class ODEBuilder:
    def __init__(self, rsc_t=None, **kwargs):
        self._slice_factories = {}
        if rsc_t is not None:
            rsc_t.register(self)
        for r in collect_quantities_(**kwargs):
            r.register(self)

    @classmethod
    def setup_ode(cls, t, derivative_computations,
                  env=None, ctrl_callback=None, ode_builder=None,
                  f_ctrl_get=None,
                  src_callback=None,
                  **kwargs):
        dx_env = {}
        ode_builder = ode_builder or cls()
        computations = collect_computations(collect_quantities_(
            derivative_computations, ctrl_callback=ctrl_callback))
        dependencies = {computation: computation.computation_dependencies()
                        for computation in computations}
        computation_jobs = depcalc.depcalc(dependencies)
        code_lines = ode_builder.write_ode_code_by_args(
            computation_jobs,
            derivative_computations,
            dx_env=dx_env,
            ctrl_callback=ctrl_callback,
            f_ctrl_get=f_ctrl_get,
            src_callback=src_callback,
            **kwargs)
        ode_builder.allocate_resources(dx_env)
        if env is not None:
            dx_env.update(env)
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
    def monitor_call(cls, computed_quantities,
                     env=None, ode_builder=None):
        ode_builder = ode_builder or cls()
        _env = {}
        named_lengths = tuple((r, r.n) for r in computed_quantities)
        out_slices = tuple(structured_data.xtuple_len_to_slc(
            named_lengths, flatten_slice=True))
        expr_slices = ((r.expression(ode_builder), slc)
                       for r, slc in out_slices)
        computations = collect_computations(
            collect_resources(computed_quantities))
        dependencies = {computation: computation.computation_dependencies()
                        for computation in computations}
        computation_jobs = depcalc.depcalc(dependencies)
        code_lines = ode_builder.write_monitor_code_by_args(
            computation_jobs, env=_env, out_slices=expr_slices)
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


def collect_quantities_from_functions(functions):
    if not functions:
        return set()
    args = set.union(*(set(f.dependencies()) for f in functions))
    return collect_resources(args)


def collect_quantities_(functions=None, ctrl_callback=None, computed_quantities=None):
    resources = set()
    if functions:
        resources |= collect_quantities_from_functions(
            functions)
    if ctrl_callback is not None:
        resources |= collect_quantities_from_functions(
            (ctrl_callback,))
    if computed_quantities:
        resources |= collect_resources(computed_quantities)
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


class QuantityContainerMeta(type):
    __register_types__ = (Quantity,
                          DerivativeBase,
                          Node)

    def __new__(mcs, name, bases, classdict):
        register = {}
        for type_ in mcs.__register_types__:
            _dct = {}
            for base in bases:
                _dct.update(base.__register__.get(type_, {}))
            _dct.update({key: val for key, val in classdict.items()
                         if isinstance(val, type_)})
            register[type_] = _dct
        classdict["__register__"] = register

        return type.__new__(mcs, name, bases, classdict)


class QuantityContainer(metaclass=QuantityContainerMeta):
    def __new__(cls, *args, name=None, parent_names=(), **kwargs):
        instance = super().__new__(cls)
        instance.__register__ = {type_: subregister.copy()
                                 for type_, subregister in cls.__register__.items()}
        instance.name = name
        instance.parent_names = parent_names
        return instance

    def __init__(self, *args, **kwargs):
        for n_name, n in self.__register__[Node].items():
            n.bind_to_parent(self, n_name)

    def __repr__(self):
        _repr = self.name or super().__repr__()
        return ".".join(tuple(self.parent_names) + (_repr,))

    def __setattr__(self, key, val):
        for type_, subregister in self.__register__.items():
            if isinstance(val, type_):
                subregister[key] = val
        super().__setattr__(key, val)
        if isinstance(val, Node):
            val.bind_to_parent(self, key)

    def __delattr__(self, key):
        super().__delattr__(key)
        for subregister in self.__register__.values():
            if key in subregister:
                del subregister[key]


class Element(QuantityContainer):
    pass


class ElementContainerMeta(QuantityContainerMeta):
    __register_types__ = QuantityContainerMeta.__register_types__ + (Element,)


class ElementContainer(QuantityContainer, metaclass=ElementContainerMeta):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)

        for n, e in instance.__register__[Element].items():
            e.name = n
        return instance


class Model(ElementContainer):
    t = ExternalQuantity(alias="t", n=1)

    def __init__(self, **kwargs):
        super().__init__()
        self.add(**kwargs)
        self.run = self._run

    def add(self, **kwargs):
        for key, arg in kwargs.items():
            setattr(self, key, arg)

    @staticmethod
    def quantities_by_type(type_):
        def _quantities(e):
            return (q for q in e.__register__[Quantity].values()
                    if isinstance(q, type_))
        return _quantities

    @staticmethod
    def _derivatives(e):
        return e.__register__[DerivativeBase].values()

    def derivatives(self):
        return self.collect_element_quantities(
            self, self._derivatives)

    def computed_quantities(self):
        return self.collect_element_quantities(
            self, self.quantities_by_type(ComputedQuantity))

    @classmethod
    def collect_element_quantities(cls, e, type_, out=None):
        if out is None:
            out = set()
        out |= set(type_(e))
        for child in e.__register__.get(Element, {}).values():
            cls.collect_element_quantities(child, type_, out=out)
        return out

    def setup_ode(self, f_ctrl=None, block_ctrl_args=None, **kwargs):
        ctrl_callback = f_ctrl and CtrlFunction(f_ctrl)
        # .computation(self, block_args=block_ctrl_args)
        derivatives = self.derivatives()
        return ODEBuilder.setup_ode(self.t, derivatives,
                                    ctrl_callback=ctrl_callback,
                                    **kwargs)

    def monitor_call(self, monitored_arguments=None, **kwargs):
        if monitored_arguments is None:
            monitored_resources = self.computed_quantities()
        else:
            monitored_resources = tuple(
                getattr_recursive(self, arg) for arg in monitored_arguments)
            wrong_args = tuple(arg for arg, r in zip(monitored_arguments, monitored_resources)
                               if not isinstance(r, ComputedQuantity))
            if wrong_args:
                raise ValueError("Arguments {} are not of type ComputedQuantity".format(
                    ", ".join(".".join(arg) for arg in wrong_args)))
        return ODEBuilder.monitor_call(monitored_resources, **kwargs)

    def compile(self, f_ctrl=None):
        sources = timeseries_interpolant()
        ode_args = {"src_callback": sources}
        if f_ctrl is not None:
            ctrl_resources = collect_quantities_from_functions((
                CtrlFunction(f_ctrl),))
            ctrl = controller()
            ode_args["f_ctrl_get"] = ctrl
            ode_args["f_ctrl"] = f_ctrl
        else:
            ctrl_resources = set()
        ode_builder = ODEBuilder(
            rsc_t=self.t,
            functions=self.derivatives(),
            computed_quantities=ctrl_resources | self.computed_quantities())
        d_X = self.setup_ode(
            ode_builder=ode_builder, **ode_args)
        Y = self.monitor_call(
            ode_builder=ode_builder)
        return d_X, Y

    def _run(self, initial_data, t, f_ctrl=None, **kwargs):
        d_X, Y = self.compile(f_ctrl=f_ctrl)

        return simulate(initial_data, t, d_X, Y=Y, **kwargs)

    @classmethod
    def run(cls, *args, **kwargs):
        return cls()._run(*args, **kwargs)


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
            raise RuntimeError('scipy.odeint failed: ' + res['message'])
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
        u0 = arrange_data(initial_data, u_lengths)
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
    if tcrit is not None:
        t_full = numpy.unique(numpy.concatenate((t, tcrit)))
        if t.size < t_full.size:
            t = t_full
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


def tuplify(x):
    if isinstance(x, tuple):
        return x
    return (x,)
