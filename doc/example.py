#!/bin/env python3

import numpy

from lib.model import Model, ExternalQuantity, ConstantQuantity, t_sampler
from tutorial.elements import Pump, MixingValve, Tank
from tutorial.ctrl import ctrl as _ctrl


model = Model(
    T_1=ConstantQuantity(10.),
    T_2=ExternalQuantity("T_2", n=1),
    pump=Pump(name="pump"))
model.add(valve=MixingValve(name="valve", T_1=model.T_1,
                            T_2=model.T_2))
model.add(tank=Tank(name="tank", T_in=model.valve.T_out,
                    Phi=model.pump.Phi, V=1.0))

# import sys
# from lib.model import Model, Element, StateFunction, ManipulatedVariable, \
#     ExternalQuantity, ComputedQuantity, GlobalExternalQuantity, \
#     Derivative, DerivativeInline, ConstantQuantity, d_, t_sampler

# model = Model.from_file("tutorial/model.json", namespace=sys.modules[__name__])

# from tutorial import elements
# model = Model.from_file("tutorial/model.json", namespace=elements)

ctrl = _ctrl(model)


x0 = {model.tank.T: 20., model.pump.Phi: 0., model.valve.position: 0.}
t = numpy.arange(60.)
_T_2 = 8. + 6. / t[-1] * t
# from lib import csv_io
# from collections import OrderedDict
# csv_io.save_csv_cols(OrderedDict(
#     (("t", t), ("T_2", _T_2))), "tutorial/sources.csv")
timeseries = model.run(x0, t, external_quantity_data={model.t: t, model.T_2: _T_2},
                       f_ctrl=ctrl, t_ctrl_sampler=t_sampler(1.))

import matplotlib.pyplot as plt
for caption, (t, y) in timeseries.items():
    plt.plot(t, y, label=caption)
plt.legend()
plt.show()
