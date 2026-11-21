#!/usr/bin/env python3
"""release-manifest.json: every artifact of a release, which installers (the GUI's one-click installs) read
to choose what to install and to check what they download.

    release_manifest.py VERSION DIR [--image TAG=DIGEST:VARIANT[:PLATFORM,...]]... [--interfaces FILE]
                        [--repository OWNER/NAME] [--commit SHA] > release-manifest.json

DIR holds the release's files with a sidecar NAME.json each (docker/archive.sh writes them: kind, name,
variant, platform, sha256, size, glibc, ...). Each --image adds a container image by tag and digest.
--interfaces is `mallard version --format json` of the release, whose interface versions the manifest
carries. Every file's SHA-256 and size are checked against the file itself.
"""
import argparse
import hashlib
import json
import os
import sys

SCHEMA = "mallard-release-manifest"
VERSION = 1
KINDS = ("archive", "sif", "wheel", "image")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_artifacts(directory, base_url):
    out = []
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".json") or entry == "release-manifest.json":
            continue
        with open(os.path.join(directory, entry)) as f:
            a = json.load(f)
        if a.get("kind") not in KINDS or a.get("name") != entry[: -len(".json")]:
            raise SystemExit(f"{entry}: not an artifact sidecar (kind {a.get('kind')!r}, name {a.get('name')!r})")
        path = os.path.join(directory, a["name"])
        if not os.path.exists(path):
            raise SystemExit(f"{entry}: {a['name']} is missing")
        actual = sha256(path)
        if a.get("sha256") != actual:
            raise SystemExit(f"{a['name']}: SHA-256 {actual}, its sidecar says {a.get('sha256')}")
        a["size"] = os.path.getsize(path)
        a["url"] = f"{base_url}/{a['name']}"
        out.append(a)
    return out


def image_artifact(spec):
    """TAG=DIGEST:VARIANT[:PLATFORM,...], e.g. ghcr.io/o/m:0.12.0=sha256:ab..:cpu:linux-x86_64,linux-aarch64."""
    tag, sep, rest = spec.partition("=")
    parts = rest.split(":")
    if not sep or len(parts) < 3 or parts[0] != "sha256" or len(parts[1]) != 64:
        raise SystemExit(f"--image {spec}: TAG=sha256:<64 hex>:VARIANT[:PLATFORM,...]")
    a = {"kind": "image", "name": tag, "variant": parts[2], "digest": f"sha256:{parts[1]}"}
    if len(parts) > 3:
        a["platforms"] = parts[3].split(",")
    return a


def manifest(version, directory, images=(), interfaces=None, repository="MatthewBonanni/mallard", commit=""):
    base_url = f"https://github.com/{repository}/releases/download/v{version}"
    artifacts = file_artifacts(directory, base_url) + [image_artifact(s) for s in images]
    m = {"schema": SCHEMA, "version": VERSION, "mallard": version}
    if commit:
        m["commit"] = commit
    if interfaces is not None:
        m["interfaces"] = interfaces.get("interfaces", interfaces)
    m["artifacts"] = sorted(artifacts, key=lambda a: (KINDS.index(a["kind"]), a["name"]))
    return m


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("version")
    p.add_argument("directory")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--interfaces")
    p.add_argument("--repository", default="MatthewBonanni/mallard")
    p.add_argument("--commit", default="")
    args = p.parse_args(argv)
    interfaces = None
    if args.interfaces:
        with open(args.interfaces) as f:
            interfaces = json.load(f)
    m = manifest(args.version, args.directory, args.image, interfaces, args.repository, args.commit)
    json.dump(m, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
