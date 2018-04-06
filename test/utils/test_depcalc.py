import unittest
from DynamicalSystemKit.utils import depcalc


class TestDepCalc(unittest.TestCase):

    def setUp(self):
        self.dep_dicts = (
            {
                "a": {"b1", "b2", "b3"},
                "b1": {"c1", "c2"},
                "a2": {"c1", "b2"}
            },
            {
                "a": {"b1", "b2", "b3"},
                "b1": {"c1", "c2"},
                "a2": {"c1", "b2"},
                "z": {"a"},
            },
            {
                "a": set(),
                "b": {"a"},
            },
        )

    def test_depcalc(self):

        for dep_dict in self.dep_dicts:
            all_nodes = set(dep_dict.keys())
            for d in dep_dict.values():
                all_nodes |= d

            dep_dict_original = dep_dict.copy()

            order = depcalc.depcalc(dep_dict)
            self.assertEqual(dep_dict, dep_dict_original)
            self.assertEqual(set(order), all_nodes)
            self.assertEqual(len(order), len(all_nodes))
            loaded = set()
            for i in order:
                loaded.add(i)
                deps = dep_dict.get(i, set())
                self.assertFalse(deps - loaded)

        dep_dict = {
            "a": {"b1", "b2", "b3"},
            "b1": {"c1", "c2"},
            "a2": {"c1", "b2"},
            "c2": {"a"}
        }
        with self.assertRaises(ValueError) as cm:
            depcalc.depcalc(dep_dict)
        nodes = set(str(cm.exception).split('{')[1].replace(
            "'", "").replace(" ", "").strip('}').split(','))
        self.assertEqual(nodes, {"a", "c2", "b1"})

    def test_deporder(self):
        for dep_dict in self.dep_dicts:
            all_nodes = set(dep_dict.keys()) | set.union(
                *dep_dict.values())

            dep_dict_original = dep_dict.copy()

            order = tuple(depcalc.deporder(all_nodes, dep_dict))
            self.assertEqual(dep_dict, dep_dict_original)
            self.assertEqual(set(order), all_nodes)
            self.assertEqual(len(order), len(all_nodes))
            loaded = set()
            for i in order:
                loaded.add(i)
                deps = dep_dict.get(i, set())
                self.assertFalse(deps - loaded)


if __name__ == '__main__':
    unittest.main()
