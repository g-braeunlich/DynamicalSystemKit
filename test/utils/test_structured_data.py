import unittest
import numpy

from DynamicalSystemKit.utils import structured_data


class TestStructuredData(unittest.TestCase):

    def test_xtuple_len_to_view(self):
        dat = numpy.arange(15.).reshape(3, 5)
        named_lengths = (("a", 1), ("b", 2), ("c", 2))
        n_v = tuple(structured_data.xtuple_len_to_view(named_lengths, dat))
        self.assertEqual(tuple(n for n, _ in n_v), ("a", "b", "c"))
        numpy.testing.assert_array_equal(
            n_v[0][1], dat[:, 0, None])
        numpy.testing.assert_array_equal(
            n_v[1][1], dat[:, 1:3])
        numpy.testing.assert_array_equal(
            n_v[2][1], dat[:, 3:5])
        named_lengths = (("a", 1), ("b", 2))
        n_v = tuple(structured_data.xtuple_len_to_view(
            named_lengths, dat, axis=0))
        self.assertEqual(tuple(n for n, _ in n_v), ("a", "b"))
        numpy.testing.assert_array_equal(
            n_v[0][1], dat[0, None])
        numpy.testing.assert_array_equal(
            n_v[1][1], dat[1:])

    def test_subdivide_named_lengths(self):
        x = (('building.T', 1), ('interior.T', 1), ('tnk_DHW.T', 5),
             ('tnk_buffer.T', 1), ('tnk_WEZ.T', 1), ('tnk_heat.T', 1))
        sub_fields = frozenset(("interior.T", "tnk_buffer.T"))
        self.assertEqual(structured_data.subdivide_named_lengths(x, sub_fields),
                         (
                             [('interior.T', 1), ('tnk_buffer.T', 1)],
                             [('building.T', 1), ('tnk_DHW.T', 5),
                              ('tnk_WEZ.T', 1), ('tnk_heat.T', 1)]
        ))

    def test_slice_mapping(self):
        header_src = (
            ("a", 1), ("b", 3), ("c", 2), ("d", 2))
        header_dest = (
            ("a", 1), ("b", 3), ("c", 2), ("d", 2))

        def slc_to_tuple(slc):
            if isinstance(slc, slice):
                return (slc.start, slc.stop, slc.step)
            return (slc, slc + 1, None)

        def tpl_slc_to_tuple(tp):
            return (slc_to_tuple(tp[0]), slc_to_tuple(tp[1]))
        slc_pairs = set(map(tpl_slc_to_tuple, structured_data.slice_mapping(
            header_src, header_dest)))
        self.assertEqual(slc_pairs, {((0, 8, None), (0, 8, None))})
        slc_pairs = set(map(tpl_slc_to_tuple, structured_data.slice_mapping(
            header_src, header_dest,
            name_mapping={"c": "d", "d": "c"})))
        self.assertEqual(slc_pairs, {((0, 4, None), (0, 4, None)),
                                     ((4, 6, None), (6, 8, None)),
                                     ((6, 8, None), (4, 6, None))})

    @staticmethod
    def test_index_list_from_named_lengths():
        lst = structured_data.index_list_from_named_lengths(
            (("a", 1), ("b", 3), ("c", 2), ("d", 2)),
            ("b", "d"))
        numpy.testing.assert_array_equal(lst, numpy.array([1, 2, 3, 6, 7]))

    @staticmethod
    def test_data_mapping():
        header_src = (
            ("a", 1), ("b", 3), ("c", 2), ("d", 2))
        header_dest = (
            ("a", 1), ("b", 3), ("c", 2), ("d", 2))

        x = numpy.arange(8.)
        y = numpy.full(8, -1.)
        structured_data.data_mapping(header_src, header_dest)(x, y)
        numpy.testing.assert_array_equal(x, y)
        numpy.testing.assert_array_equal(
            x, structured_data.data_mapping(header_src, header_dest)(x))

        y = numpy.full(8, -1.)
        structured_data.data_mapping(header_src, header_dest, name_mapping={
                                     "c": "d", "d": "c"})(x, y)
        numpy.testing.assert_array_equal(
            y, numpy.array([0., 1., 2., 3., 6., 7., 4., 5.]))

        y = numpy.full(5, -1.)
        header_dest = (
            ("b", 3), ("d", 2))
        structured_data.data_mapping(header_src, header_dest)(x, y)
        numpy.testing.assert_array_equal(
            y, numpy.array([1., 2., 3., 6., 7.]))

    @staticmethod
    def test_data_mapping_n__1():
        x1 = numpy.arange(3.)
        x2 = numpy.arange(3., 8.)
        y = numpy.full(8, -1.)

        header_src_1 = (
            ("a", 1), ("c", 2))
        header_src_2 = (
            ("b", 3), ("d", 2))
        header_dest = (
            ("b", 3), ("c", 2), ("a", 1), ("d", 2))
        structured_data.data_mapping_n__n(
            (header_src_1, header_src_2), (header_dest,))(x1, x2, y)
        numpy.testing.assert_array_equal(
            y, numpy.array([3., 4., 5., 1., 2., 0., 6., 7.]))

    @staticmethod
    def test_data_mapping_n__n():
        x1 = numpy.arange(3.)
        x2 = numpy.arange(3., 8.)
        y1 = numpy.full(5, -1.)
        y2 = numpy.full(3, -1.)

        header_src_1 = (
            ("a", 1), ("c", 2))
        header_src_2 = (
            ("b", 3), ("d", 2))
        header_dest_1 = (
            ("b", 3), ("c", 2))
        header_dest_2 = (
            ("a", 1), ("d", 2))
        structured_data.data_mapping_n__n(
            (header_src_1, header_src_2), (header_dest_1, header_dest_2))(x1, x2, out0=y1, out1=y2)
        numpy.testing.assert_array_equal(
            y1, numpy.array([3., 4., 5., 1., 2.]))
        numpy.testing.assert_array_equal(
            y2, numpy.array([0., 6., 7.]))

    def test_slc_complement(self):
        slc = slice(1, 3)
        numpy.testing.assert_array_equal(structured_data.slc_complement(
            slc, 4), numpy.array([True, False, False, True]))
        slc = slice(1, 3)
        self.assertEqual(structured_data.slc_complement(
            slc, 3), numpy.array(slice(None, 1)))
        slc = numpy.array([True, False, False, True])
        numpy.testing.assert_array_equal(structured_data.slc_complement(
            slc, 4), numpy.array([False, True, True, False]))
        slc = numpy.array([True, False, False])
        self.assertEqual(structured_data.slc_complement(
            slc, 3), slice(1, None))

    def test_slc_compose(self):
        array = tuple(range(10))
        array2 = tuple(range(11))
        indices_inner = (-2, -1, 0, None, 1, 3)
        indices_outer = (0, None, 1, 3)
        # steps = (-2, -1, None, 1, 3)
        steps = (None, 1, 3)
        slices_inner = tuple(slice(a, b, step)
                             for a in indices_inner for b in indices_inner for step in steps)
        slices_outer = tuple(slice(a, b, step)
                             for a in indices_outer for b in indices_outer for step in steps)
        slice_tuples = ((outer, inner)
                        for outer in slices_outer for inner in slices_inner)
        # slice_tuples = ((slice(1, 3, 3), slice(3, -1, 3)),
        #                 (slice(3, -1, 3), slice(1, 3, 3)))
        for outer, inner in slice_tuples:
            with self.subTest(outer=outer, inner=inner):
                try:
                    result = array[outer][inner]
                    result2 = array2[outer][inner]
                    slc = structured_data.slc_compose(outer, inner)
                    res1 = (result == array[slc])
                    res2 = (result2 == array2[slc])
                    if res1 != res2:
                        raise ValueError("Undefined behaviour")
                    self.assertEqual(result, array[slc])
                    self.assertEqual(result2, array2[slc])
                except (IndexError, ValueError) as err:
                    with self.assertRaises(type(err)):
                        structured_data.slc_compose(outer, inner)


if __name__ == '__main__':
    unittest.main()
