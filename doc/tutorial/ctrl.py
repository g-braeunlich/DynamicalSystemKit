from math import sin


def ctrl(__model__):
    def _ctrl(t: __model__.t, T: __model__.tank.T,
              Phi: __model__.pump.Phi[:],
              position: __model__.valve.position[:]):
        Phi[:] = 0.5 if T < 10. else 1.0
        position[:] = sin(t)
    return _ctrl
