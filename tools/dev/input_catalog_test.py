#!/usr/bin/env python3
"""Tests of the merge of tools/dev/input_catalog.py build (run by tools/dev/precheck.sh)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import input_catalog  # noqa: E402


def merged(*catalogs):
    out, untyped = {}, set()
    for i, keys in enumerate(catalogs):
        input_catalog.merge(out, keys, 2, f"input{i}.toml", untyped)
    input_catalog.finish(out, untyped)
    return out


class Merge(unittest.TestCase):
    def test_types_merge_unless_an_input_reads_the_key_without_one(self):
        typed = {"p.radius": {"set": True, "type": "number", "required": True, "for_types": ["circle"],
                              "required_for": ["circle"]}}
        other = {"p.radius": {"set": False, "type": "number", "default": "1.0", "for_types": ["arc"]}}
        out = merged(typed, other)["p.radius"]
        self.assertEqual(out["for_types"], ["arc", "circle"])
        self.assertEqual(out["required_for"], ["circle"])
        untyped = {"p.radius": {"set": True, "type": "number", "required": True}}
        out = merged(typed, untyped)["p.radius"]
        self.assertNotIn("for_types", out)
        self.assertNotIn("required_for", out)

    def test_alternatives_are_required_only_if_every_input_requires_them(self):
        a = {"t": {"set": True, "alternatives": [{"keys": [["X"], ["Y"]], "required": True}]}}
        b = {"t": {"set": True, "alternatives": [{"keys": [["X"], ["Y"]], "required": False},
                                                 {"keys": [["dt"], ["cfl"]], "required": True}]}}
        self.assertEqual(merged(a)["t"]["alternatives"], [{"keys": [["X"], ["Y"]], "required": True}])
        self.assertEqual(merged(a, b)["t"]["alternatives"], [{"keys": [["X"], ["Y"]], "required": False},
                                                             {"keys": [["dt"], ["cfl"]], "required": True}])

    def test_paths_and_references_carry_through(self):
        keys = {"m.file": {"set": True, "type": "string", "path": {"io": "input", "kind": "file"}},
                "m.zone": {"set": True, "type": "string", "references": "zones"}}
        out = merged(keys)
        self.assertEqual(out["m.file"]["path"], {"io": "input", "kind": "file"})
        self.assertEqual(out["m.zone"]["references"], "zones")

    def test_models_merge_over_the_inputs(self):
        # Each input's setup records the models it lists; the catalog keeps those of every input
        a = {"les.filter_width": {"set": False, "models": {"scotti": {"docs": "docs/theory/les.md#filter-width"}}}}
        b = {"les.filter_width": {"set": False, "models": {"volume": {"docs": "docs/theory/les.md#filter-width"}}}}
        self.assertEqual(set(merged(a, b)["les.filter_width"]["models"]), {"scotti", "volume"})


if __name__ == "__main__":
    unittest.main()
