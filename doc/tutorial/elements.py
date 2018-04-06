from sim.model import Model, Element, StateFunction, ManipulatedVariable, \
    ExternalQuantity, ComputedQuantity, GlobalExternalQuantity, \
    Derivative, DerivativeInline, ConstantQuantity, d_, t_sampler


class Pump(Element):
    Phi = ManipulatedVariable(1)


class MixingValve(Element):
    position = ManipulatedVariable(1)

    def __init__(self, *args, T_1, T_2, **kwargs):
        super().__init__(*args, **kwargs)

        @ComputedQuantity
        def T_out(T1: T_1, T2: T_2, position: self.position) -> 1:
            return T1 * position + T2 * (1. - position)
        self.T_out = T_out


class Tank(Element):
    T = StateFunction(1)

    def __init__(self, *args, T_in, Phi, V, **kwargs):
        super().__init__(*args, **kwargs)

        @Derivative
        def dT(T: self.T, Phi_: Phi, T_i: T_in) -> d_(self.T):
            return Phi_ / V * (T_i - T)
        self.dT = dT
