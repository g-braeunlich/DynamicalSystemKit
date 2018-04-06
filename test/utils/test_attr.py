import unittest

from DynamicalSystemKit.utils import attr


class B:

    def __init__(self):
        self.c = 1


class A:

    def __init__(self):
        self.b = B()


class Node:
    pass


class TestUtilsAttr(unittest.TestCase):

    def test_getattr_recursive(self):
        a = A()
        c = attr.getattr_recursive(a, ("b", "c"))
        self.assertTrue(hasattr(a, "b"))
        self.assertTrue(hasattr(a.b, 'c'))
        self.assertEqual(c, 1)

    def test_setattr_recursive(self):
        n = Node()
        attr.setattr_recursive(n, ("b", "c"), 1)
        self.assertTrue(hasattr(n, 'b'))
        self.assertTrue(hasattr(n.b, 'c'))
        self.assertEqual(n.b.c, 1)


if __name__ == '__main__':
    unittest.main()
