#!/usr/bin/env python3
"""Tests of release_manifest.py: what installers read must name every artifact with a SHA-256 that matches
the file, and refuse sidecars that do not."""
import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_manifest  # noqa: E402


def write_archive(d, name, data=b"archive bytes", **fields):
    with open(os.path.join(d, name), "wb") as f:
        f.write(data)
    sidecar = {"kind": "archive", "name": name, "variant": "cpu", "platform": "linux-x86_64",
               "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "glibc": "2.28"}
    sidecar.update(fields)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(sidecar, f)


class ReleaseManifestTest(unittest.TestCase):
    def test_archives_and_images_with_urls_digests_and_interfaces(self):
        with tempfile.TemporaryDirectory() as d:
            write_archive(d, "mallard-0.12.0-linux-x86_64-cpu.tar.gz")
            write_archive(d, "mallard-0.12.0-linux-aarch64-cpu.tar.gz", b"other", platform="linux-aarch64")
            digest = "ab" * 32
            m = release_manifest.manifest(
                "0.12.0", d, [f"ghcr.io/matthewbonanni/mallard:0.12.0=sha256:{digest}:cpu:linux-x86_64,linux-aarch64"],
                {"schema": "mallard-version", "interfaces": {"run_log": 3}}, commit="abc")
        self.assertEqual((m["schema"], m["version"], m["mallard"], m["commit"]),
                         ("mallard-release-manifest", 1, "0.12.0", "abc"))
        self.assertEqual(m["interfaces"], {"run_log": 3})
        names = [a["name"] for a in m["artifacts"]]
        self.assertEqual(names, ["mallard-0.12.0-linux-aarch64-cpu.tar.gz", "mallard-0.12.0-linux-x86_64-cpu.tar.gz",
                                 "ghcr.io/matthewbonanni/mallard:0.12.0"])
        x86 = m["artifacts"][1]
        self.assertEqual(x86["url"], "https://github.com/MatthewBonanni/mallard/releases/download/v0.12.0/"
                                     "mallard-0.12.0-linux-x86_64-cpu.tar.gz")
        self.assertEqual(x86["sha256"], hashlib.sha256(b"archive bytes").hexdigest())
        self.assertEqual(m["artifacts"][2]["digest"], f"sha256:{digest}")
        self.assertEqual(m["artifacts"][2]["platforms"], ["linux-x86_64", "linux-aarch64"])

    def test_a_file_that_does_not_match_its_sidecar_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            write_archive(d, "mallard-0.12.0-linux-x86_64-cpu.tar.gz", sha256="0" * 64)
            with self.assertRaisesRegex(SystemExit, "its sidecar says"):
                release_manifest.manifest("0.12.0", d)
            os.remove(os.path.join(d, "mallard-0.12.0-linux-x86_64-cpu.tar.gz"))
            with self.assertRaisesRegex(SystemExit, "is missing"):
                release_manifest.manifest("0.12.0", d)

    def test_an_image_needs_its_digest(self):
        with self.assertRaisesRegex(SystemExit, "sha256:<64 hex>"):
            release_manifest.image_artifact("ghcr.io/matthewbonanni/mallard:0.12.0=latest:cpu")


if __name__ == "__main__":
    unittest.main()
