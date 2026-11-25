#!/usr/bin/env python3
"""Tests of tools/dev/check_subview_copies.py (run by tools/dev/precheck.sh)."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_subview_copies  # noqa: E402


def flagged(body):
    """The lines of body (inside a function of views v (default layout), l (LayoutLeft) and r
    (LayoutRight, rank 3), with range a pair) that the check flags."""
    head = ("void f(const Kokkos::View<double **> & v, Kokkos::View<double **, Kokkos::LayoutLeft> l,\n"
            "       Kokkos::View<float ***, Kokkos::LayoutRight> r, size_t n) {\n"
            "    const std::pair<size_t, size_t> range{0, n};\n")
    with tempfile.TemporaryDirectory() as root:
        path = pathlib.Path(root, "a.cpp")
        path.write_text(head + body + "}\n")
        problems = check_subview_copies.check(str(path), root)
    return [int(p.split(":")[1]) - 3 for p in problems]


class SubviewCopies(unittest.TestCase):
    def test_row_ranges_of_default_layout_views_to_the_host(self):
        # The copies of #1418 and #1395 that threw on CUDA
        self.assertEqual(flagged(
            "    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, range, Kokkos::ALL));\n"
            "    auto h_b = Kokkos::create_mirror_view(v);\n"
            "    Kokkos::deep_copy(h_b, Kokkos::subview(v, range, Kokkos::ALL));\n"
            "    Kokkos::deep_copy(Kokkos::subview(v, range, Kokkos::ALL), HostUnmanaged<double>(p, n));\n"), [1, 3, 4])

    def test_contiguous_in_the_explicit_layout(self):
        self.assertEqual(flagged(
            "    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(l, Kokkos::ALL, 3));\n"
            "    auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(l, 3, Kokkos::ALL));\n"
            "    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(r, 2, range, Kokkos::ALL));\n"
            "    auto d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(r, range, 1, Kokkos::ALL));\n"
            "    auto e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, 1, 2));\n"),
            [2, 4])

    def test_device_copies_and_marked_copies_pass(self):
        self.assertEqual(flagged(
            "    Kokkos::View<double **> dev(\"dev\", n, 3);\n"
            "    Kokkos::deep_copy(dev, Kokkos::subview(v, range, Kokkos::ALL));\n"
            "    // subview-copy: contiguous (one column)\n"
            "    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, range, Kokkos::ALL));\n"),
            [])


if __name__ == "__main__":
    unittest.main()
