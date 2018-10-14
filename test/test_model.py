import unittest
import numpy

from DynamicalSystemKit import model
from DynamicalSystemKit.model import d_
from DynamicalSystemKit.model import Model, Element, StateFunction, ManipulatedVariable, \
    ComputedQuantity, Derivative, ConstantQuantity, arrange_data


class MockParent:
    def __init__(self, name, parent_names=()):
        self.name = name
        self.parent_names = parent_names


class TestStateElement(model.Element):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = model.StateFunction(2)


class TestManipulatingElement(model.Element):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = model.ManipulatedVariable(1)


class TestComputedElement(model.Element):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = model.StateFunction(1)

        @model.ComputedQuantity
        def h(x: self.x) -> 1:
            return x
        self.h = h

        @model.Derivative
        def d_x(_h: h) -> d_(self.x):
            return _h

        self.d_x = d_x


class TestUtils(unittest.TestCase):
    def test_arrange_data(self):
        data = {"a": numpy.array([1, 1]), "b": 2, "c": 3}
        numpy.testing.assert_array_equal(
            model.arrange_data(data, (('a', 2), ('c', 1))),
            numpy.array([1, 1, 3])
        )
        with self.assertRaises(KeyError):
            model.arrange_data(data, (('a', 2), ('d', 1)))
        with self.assertRaises(ValueError):
            model.arrange_data(data, (('a', 3), ('c', 1)))
        data = {"a": numpy.array([[1, 2], [1, 2]]),
                "b": numpy.array([[3], [3]]), "c": numpy.array([4, 4])}
        numpy.testing.assert_array_equal(
            model.arrange_data(
                data, (('a', 2), ('c', 1), ('b', 1)), extra_shape=(2,)),
            numpy.array([[1, 2, 4, 3], [1, 2, 4, 3]])
        )


class TestODEFactory(unittest.TestCase):
    def test_register_derivative_computation(self):
        A_x = model.StateFunction(parent="A", alias="x")

        v = model.ExternalQuantity(alias="v", n=1)
        B_u = model.ManipulatedVariable(parent="B", alias="u")

        def h1(v: v, u: B_u) -> 1:
            pass

        B_h1 = model.ComputedQuantity(h1, parent=MockParent("B"),
                                      alias="h1")

        def h2(h1: B_h1) -> 2:
            pass

        B_h2 = model.ComputedQuantity(h2,
                                      parent=MockParent("B"),
                                      alias="h2")

        def dx(x: A_x, y: B_h2) -> d_(A_x):
            pass

        def h3(h2: B_h2) -> 1:
            pass

        def h4() -> 1:
            pass

        A_dx = model.Derivative(dx)
        A_dx.parent = MockParent("A")
        ode_builder = model.ODEBuilder()
        env = {}
        code_A_dx = A_dx.write_code(ode_builder, env)
        code_B_h1 = B_h1.function.write_code(ode_builder, env)
        code_B_h2 = B_h2.function.write_code(ode_builder, env)
        self.assertEqual(code_A_dx, "_dx[0] += A_dx(_x[0], B_h2)")
        self.assertEqual(code_B_h2, "B_h2 = f_B_h2(B_h1)")
        self.assertEqual(code_B_h1, "B_h1 = f_B_h1(_v[0], _u[0])")

    def test_register_dx_callbacks_2(self):
        A_x = model.StateFunction(parent="A", alias="x")
        A_y = model.StateFunction(parent="A", alias="y")

        def dx() -> (d_(A_x), d_(A_y)):
            pass
        A_dx = model.Derivative(dx)
        A_dx.parent = MockParent("A")

        ode_builder = model.ODEBuilder()
        env = {}
        code = A_dx.write_code(ode_builder, env)
        self.assertTrue(
            code == "_tmp0, _tmp1 = A_dx() ; _dx[0] += _tmp0 ; _dx[1] += _tmp1"
            or
            code == "_tmp0, _tmp1 = A_dx() ; _dx[1] += _tmp0 ; _dx[0] += _tmp1")

    def test_register_dx_callbacks_inline(self):
        A_x = model.StateFunction(parent="A", alias="x")
        A_y = model.StateFunction(parent="A", alias="y")

        def dx(d_x: d_(A_x), d_y: d_(A_y)):
            pass
        A_dx = model.DerivativeInline(dx)
        A_dx.parent = MockParent("A")
        ode_builder = model.ODEBuilder()
        env = {}
        code = A_dx.write_code(ode_builder, env)
        slices = ode_builder._slice_factories[model.StateFunction].slices
        self.assertEqual(set(slices.values()), {0, 1})
        slc_x = slices[A_x]
        slc_y = slices[A_y]
        self.assertEqual(code,
                         "A_dx(_dx[{}], _dx[{}])".format(slice(slc_x, slc_x + 1),
                                                         slice(slc_y, slc_y + 1)))

    def test_setup_ode_simple(self):
        A_x = model.StateFunction(parent="A", alias="x")

        def d_x(x: A_x) -> d_(A_x):
            return x + 1.

        A_dx = model.Derivative(d_x)
        dx = model.ODEBuilder.setup_ode(None, (A_dx,))
        self.assertEqual(dx(numpy.full(1, 0.), 2.), 1.)
        self.assertEqual(dx(numpy.full(1, 1.), 3.), 2.)

    def test_setup_ode_2(self):
        A_x = model.StateFunction(parent="A", alias="x")
        A_y = model.StateFunction(parent="A", alias="y")
        v_ext = model.ExternalQuantity(alias="v", n=1)
        B_u = model.ManipulatedVariable(parent="B", alias="u")

        def h1(v: v_ext, u: B_u) -> 1:
            self.assertEqual(v, 6.)
            self.assertEqual(u, 7.)
            return 8.

        B_h1 = model.ComputedQuantity(h1,
                                      parent=MockParent("B"), alias="h1")

        def h2(h1: B_h1) -> 2:
            self.assertEqual(h1, 8.)
            return numpy.array([2., 3.])
        A_y = model.ComputedQuantity(h2,
                                     parent=MockParent("B"), alias="h2")

        def d_x(x: A_x, y: A_y) -> d_(A_x):
            self.assertEqual(x, 1.)
            numpy.testing.assert_array_equal(y, numpy.array([2., 3.]))
            return 4.

        A_dx = model.Derivative(d_x)

        def d_x2() -> d_(A_x):
            return 5.
        A_dx2 = model.Derivative(d_x2)

        derivative_computations = (A_dx, A_dx2)

        def ctrl_get(t):
            self.assertEqual(t, ctrl_get.t_ref)
            ctrl_get.t_ref = -1.1
            return numpy.full(1, 7.), t == -1.1
        ctrl_get.t_ref = -1.

        ctrl_called = False

        @model.CtrlFunction
        def ctrl():
            nonlocal ctrl_called
            ctrl_called = True

        def src_callback(t, v):
            v[:] = 6.
        dx = model.ODEBuilder.setup_ode(
            None,
            derivative_computations,
            f_ctrl_get=ctrl_get, ctrl_callback=ctrl,
            src_callback=src_callback)
        numpy.testing.assert_array_equal(dx(numpy.full(2, 1.), -1.), 9.)
        self.assertFalse(ctrl_called)
        numpy.testing.assert_array_equal(dx(numpy.full(2, 1.), -1.1), 9.)
        self.assertTrue(ctrl_called)

    def test_model(self):
        class TestConditionalElement(model.Element):
            def __init__(self, name, e, h, **kwargs):
                super().__init__(name, **kwargs)
                self.e = e
                self.h = h

                @model.DerivativeInline
                def d_x(h: self.h, dx: d_(self.e.x)):
                    dx[1] = h
                self.d_x = d_x

        model_args = {"e_x": TestStateElement(name="e_x"),
                      "e_u": TestManipulatingElement(name="e_u")}
        model_args["e_h"] = TestConditionalElement(
            e=model_args["e_x"],
            name="e_h",
            h=model_args["e_u"].u)
        mod = model.Model(**model_args)
        self.assertEqual(mod.e_h.e, mod.e_x)
        self.assertEqual(mod.e_h.h, mod.e_u.u)
        a1, a2 = mod.e_h.d_x.arguments
        self.assertEqual(a1, mod.e_h.h)
        self.assertEqual(a2.quantity, mod.e_h.e.x)

    def test_model_2(self):
        model_args = {"e": TestComputedElement(name="e")}
        mod = model.Model(**model_args)
        self.assertEqual(mod.e.d_x.arguments, (mod.e.h,))
        self.assertEqual(mod.e.h.function.arguments, (mod.e.x,))

    def test_model_3(self):
        class Pump(Element):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Phi = ManipulatedVariable(1)

        class MixingValve(Element):
            def __init__(self, *args, T_1, T_2, **kwargs):
                super().__init__(*args, **kwargs)
                self.position = ManipulatedVariable(1)

                @ComputedQuantity
                def T_out(T1: T_1, T2: T_2, position: self.position) -> 1:
                    return T1 * position + T2 * (1. - position)
                self.T_out = T_out

        class Tank(Element):
            def __init__(self, *args, T_in, Phi, **kwargs):
                super().__init__(*args, **kwargs)
                self.T = StateFunction(1)

                @Derivative
                def dT(T: self.T, Phi_: Phi, T_i: T_in) -> d_(self.T):
                    return Phi_ * (T_i - T)
                self.dT = dT

        T_const = 10.
        T_const_2 = 10.

        class ExampleModel(Model):
            T_1 = ConstantQuantity(T_const)
            T_2 = ConstantQuantity(T_const_2)
            pump = Pump()
            valve = MixingValve(T_1=T_1, T_2=T_2)
            tank = Tank(T_in=valve.T_out, Phi=pump.Phi)

        Phi_0 = 1.
        T_0 = 20.
        pos_0 = 0.4
        T_i = ExampleModel.valve.T_out(
            T1=T_const, T2=T_const_2, position=pos_0)
        initial_data = {ExampleModel.tank.T: T_0, ExampleModel.pump.Phi: Phi_0,
                        ExampleModel.valve.position: pos_0}
        d_X = ExampleModel().setup_ode()
        u_lengths = d_X.quantity_lengths.get(ManipulatedVariable)
        u0 = arrange_data(initial_data, u_lengths)
        d_X.u[()] = u0
        x0 = arrange_data(initial_data, d_X.quantity_lengths[StateFunction])
        numpy.testing.assert_array_equal(
            d_X(x0, 0.), numpy.array([ExampleModel.tank.dT(T=T_0, Phi_=Phi_0, T_i=T_i)]))

    def test_dynamic_quantity(self):
        class TestDynamicElement(model.Element):
            q_static = model.ConstantQuantity(1.)

            def __init__(self, name, q):
                self.q_dynamic = q

        q = model.ConstantQuantity(10.)
        e = TestDynamicElement(name="e", q=q)
        self.assertEqual(TestDynamicElement.__register__[model.Quantity], {
                         "q_static": TestDynamicElement.q_static})
        self.assertEqual(e.__register__[model.Quantity], {
                         "q_static": TestDynamicElement.q_static,
                         "q_dynamic": q})
        self.assertEqual(e.__register__[model.Quantity],
                         {
                             "q_static": TestDynamicElement.q_static,
                             "q_dynamic": q})
        delattr(e, "q_dynamic")
        self.assertEqual(e.__register__[model.Quantity],
                         {"q_static": TestDynamicElement.q_static})


class TestController(unittest.TestCase):

    def test_controller(self):
        t = numpy.arange(5.)
        u0 = numpy.array([0., 0., -1.])
        u1 = numpy.full(3, 1.)
        C = model.controller(t, u0)

        u, trigger_ctrl = C(0.)
        self.assertEqual(C.end, 1)
        self.assertEqual(trigger_ctrl, False)
        numpy.testing.assert_array_equal(u0, u)
        u, trigger_ctrl = C(0.5)
        self.assertEqual(C.end, 1)
        self.assertEqual(trigger_ctrl, False)
        numpy.testing.assert_array_equal(u0, u)
        u, trigger_ctrl = C(1.)
        self.assertEqual(C.end, 2)
        self.assertEqual(trigger_ctrl, True)
        numpy.testing.assert_array_equal(u0, u)
        u, trigger_ctrl = C(2.1)
        self.assertEqual(trigger_ctrl, True)
        self.assertEqual(C.end, 3)
        numpy.testing.assert_array_equal(u0, u)
        u, trigger_ctrl = C(0.1)
        self.assertEqual(trigger_ctrl, False)
        self.assertEqual(C.end, 3)
        numpy.testing.assert_array_equal(u0, u)
        u[:] = 0.
        u, trigger_ctrl = C(1.1)
        self.assertEqual(trigger_ctrl, False)
        self.assertEqual(C.end, 3)
        numpy.testing.assert_array_equal(u, u0)
        u, trigger_ctrl = C(2.3)
        self.assertEqual(trigger_ctrl, False)
        self.assertEqual(C.end, 3)
        numpy.testing.assert_array_equal(u0, u)
        u[:] = u1
        with self.assertWarns(UserWarning):
            u, trigger_ctrl = C(4.9)
        self.assertEqual(trigger_ctrl, True)
        numpy.testing.assert_array_equal(C.u, numpy.array(
            [[0., 0., 0.], u0, u1, u1, u1]))


if __name__ == '__main__':
    unittest.main()
