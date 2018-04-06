_no_default = object()


def getattr_recursive(obj, addr, default=_no_default):
    try:
        for part in addr:
            obj = getattr(obj, part)
    except AttributeError as err:
        if default is not _no_default:
            return default
        raise err

    return obj


def setattr_recursive(e, addr, val):
    f = e
    for a in addr[:-1]:
        if not hasattr(f, a):
            setattr(f, a, Node())
        f = getattr(f, a)
    setattr(f, addr[-1], val)


def hasattr_recursive(e, addr):
    f = e
    for a in addr:
        if not hasattr(f, a):
            return False
        f = getattr(f, a)
    return True


class Node:
    pass
