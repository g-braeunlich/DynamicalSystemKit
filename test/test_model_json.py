import unittest

from DynamicalSystemKit import model_json


class TestElements:
    obj = object()

    class Nested:
        obj = object()

    class A:
        def __init__(self, a1, a2):
            self.a1 = a1
            self.a2 = a2
            self.content = {"a1": a1, "a2": a2}

    class B:
        def __init__(self, ref, a, dct):
            self.content = {"ref": ref, "a": a, "dct": dct}

    class WithQuantity:
        def __init__(self, alpha):
            self.quantity = object()
            self.content = {"alpha": alpha}

    class WithForeignQuantity:
        def __init__(self, a, foreign):
            self.foreign = foreign
            self.content = {"a": a}


class TestModelJSON(unittest.TestCase):
    def test_model_args_from_dict_1(self):
        tree = {"a": {"__type__": "A", "a1": 1, "a2": 2},
                "b": {"__type__": "B",
                      "ref": {"__reference__": "c.a2"},
                      "a": {"__type__": "A", "a1": 11, "a2": 12},
                      "dct": {"alpha": -1, "beta": -2}},
                "c": {"__type__": "A", "a1": 21, "a2": 22}}
        converted = model_json.model_args_from_dict(tree, TestElements)

        self.assertEqual(set(converted.keys()), {"a", "b", "c"})
        self.assertEqual(converted["a"].content, {"a1": 1, "a2": 2})
        self.assertEqual(converted["c"].content, {"a1": 21, "a2": 22})
        self.assertEqual(set(converted["b"].content.keys()),
                         {"ref", "a", "dct"})
        self.assertEqual(converted["b"].content["ref"], 22)
        self.assertEqual(converted["b"].content["a"].content,
                         {"a1": 11, "a2": 12})
        self.assertEqual(converted["b"].content["dct"],
                         {"alpha": -1, "beta": -2})

    def test_model_args_from_dict_2(self):
        tree = {"a": {"__type__": "A", "a1": 1, "a2": 2},
                "b": {"ref": {"__reference__": "a"}},
                "c": {"__reference__": "b.ref"}}
        converted = model_json.model_args_from_dict(tree, TestElements)

        a = converted["a"]
        self.assertEqual(a.content, {"a1": 1, "a2": 2})
        self.assertEqual(converted, {"b": {"ref": a}, "c": a, "a": a})

    def test_model_args_from_dict_3(self):
        tree = {"ref1": {"__reference__": "ref2"},
                "ref2": {"__reference__": "ref1"}}
        with self.assertRaises(ValueError) as cm:
            model_json.model_args_from_dict(tree, TestElements)
        self.assertTrue(str(cm.exception).startswith("Circular reference"))

    def test_model_args_from_dict_4(self):
        tree = {"ref": {"__reference__": "x"}}
        with self.assertRaises(KeyError):
            model_json.model_args_from_dict(tree, TestElements)

    def test_model_args_from_dict_5(self):
        tree = {"a": {"ref": {"__reference__": "b"}},
                "b": {"ref": {"__reference__": "a"}}}
        with self.assertRaises(ValueError) as cm:
            model_json.model_args_from_dict(tree, TestElements)
        self.assertTrue(str(cm.exception).startswith("Circular dependency"))

    def test_model_args_from_dict_6(self):
        tree = {"a": 1,
                "b": {"__reference__": "a"}}
        converted = model_json.model_args_from_dict(tree, TestElements)

        a = converted["a"]
        self.assertEqual(a, 1)
        self.assertEqual(converted["b"], a)

    def test_model_args_from_dict_obj(self):
        tree = {"a": {"__object__": "Nested.obj"},
                "b": {"__object__": "obj"}}
        converted = model_json.model_args_from_dict(tree, TestElements)

        a = converted["a"]
        self.assertEqual(a, TestElements.Nested.obj)
        self.assertEqual(converted["b"], TestElements.obj)

    def test_model_args_from_dict_foreign_quantity(self):
        class MockWrapper:
            def __init__(self):
                self.target = None
        tree = {"a": {"__type__": "WithQuantity", "alpha": 1},
                "b": {"__type__": "WithForeignQuantity", "a": 2,
                      "foreign": {"__foreign_quantity__": "a.quantity"}}
                }
        converted = model_json.model_args_from_dict(tree, TestElements,
                                                    foreign_quantity_wrapper=MockWrapper)

        a = converted["a"]
        b = converted["b"]
        self.assertEqual(a.content, {"alpha": 1})
        self.assertEqual(b.content, {"a": 2})
        self.assertTrue(b.foreign.target is a.quantity)
        self.assertEqual(converted, {"b": b, "a": a})


if __name__ == '__main__':
    unittest.main()
