# Local coordinates of boundary patches

Why boundary expressions read `u`, `v` and `n`
([guide](../guide/boundary_conditions.md#local-coordinates)), what they are
on each kind of patch, and how they are computed
(`src/boundary/patch_coordinates.cpp`, `boundary_coordinates.cpp`).

## Goal

A profile such as `U * 16 * u * (1 - u) * v * (1 - v)` or `1 - r^2` should
work on any duct or pipe with no dimensions, no origin and no axes in the
input: the coordinates must come from the boundary's own shape. The first
version of the local coordinates (unreleased) used a frame instead, an
origin, a normal and a tangent fitted to the faces, with `xi`, `eta`, `r`,
`theta` in meters in that frame. It needed the dimensions of the inlet in
every expression, had no meaning on a curved patch, failed on closed
surfaces (the normals of a pipe's wall cancel), and its default tangent
could flip between two meshes of an oblique inlet. It is removed; the patch
coordinates replace it.

## The set

Every coordinate is a fraction with a bare name and has a twin in meters
with the suffix `_m`. There is one family along the boundary (`u`, `v`) and
one from the walls (`n`); the rest are other names for these.

| Name | Four-sided | Ring | Disk | Curve (2D) |
|---|---|---|---|---|
| `u` | harmonic, 0 and 1 on two opposite sides | conjugate of `v` around the ring, in [0, 1) | angle of the conformal map to the unit disk, in [0, 1) | arc length / length |
| `v` | harmonic, 0 and 1 on the other two sides | harmonic, 0 and 1 on the rims | radius of that map: 0 at the center, 1 on the edge | |
| `u_m` | distance on the surface from the side `u = 0` | `u` times the length around at the face | as the ring | arc length |
| `v_m` | distance on the surface from the side `v = 0` | distance on the surface from the rim `v = 0` | distance on the surface from the center | |
| `n`, `n_m` | wall distance | wall distance | wall distance | wall distance |
| `r`, `r_m` | error | `v`, `v_m` | `v`, `v_m` | not in 2D |
| `theta` | error | `2 pi u` | `2 pi u` | not in 2D |

Decisions:

- **`r` and `theta` are names, not new quantities.** A disk's `v` is its
  radius fraction and `2 pi u` its angle, so `1 - r^2` reads as it does in a
  textbook. On a four-sided patch they have no natural meaning and are an
  error that names the patch, rather than a square "radius" nobody expects.
- **The lengths follow the surface.** `v_m` is the geodesic distance from the
  set `v = 0`, so it is in meters on a curved panel or a cone too, and its
  level lines are those of `v` wherever the patch is regular (rectangle,
  tube, circle; on an annulus both are circles). It is not `v` times a
  length, which would be a length only on those shapes: on an annulus `v` is
  logarithmic in the radius, and a boundary layer needs true meters. Around a
  ring or a disk the surface has no side to measure from, so `u_m` is `u`
  times the length around at the face, interpolated linearly in `v_m` between
  the lengths of the two rims (0 at a disk's center): the exact arc length on
  circles, annuli, cones and tubes.
- **The wall distance stays the 3D distance to wall faces** rather than a
  distance on the patch to its edges. The walls a boundary layer grows on are
  zones of the mesh, not edges of the patch: an inlet's edges may be symmetry
  planes or other inlets, the inlet may not touch the wall it should measure
  from (`walls` names any zones), and a 2D boundary has ends only. The
  distance to the wall faces is exact (no discretization), the same in 2D
  and 3D, and equals the distance on the patch where the walls meet it at
  right angles. Its fraction `n` is `n_m` over its largest value on the
  boundary's faces and nodes, so it has exactly the level lines of `n_m`,
  and `n^(1/7)` is 1 at the middle of any duct.
- **No length constants.** The lengths of the patch along `u` and `v` are the
  upper ends of the ranges of `u_m` and `v_m` in the run log; expressions
  that need a fraction have `u` and `v`.
- **One way to say where `u` starts: two points in meters.** `origin` is
  where `u` (and `v`) is 0 and `toward` where `u` grows to, on every patch:
  the corner of a four-sided patch nearest to `origin` becomes (0, 0) and
  `u` runs to the adjacent corner nearest to `toward`; on a ring or a disk
  `u` is 0 at the face nearest to `origin` and grows the way around that
  reaches `toward` sooner; in 2D `origin` picks the end. Points are what a
  user reads off the mesh or the CAD, they need no notion of an axis (which
  a cone, a bent pipe or an oblique panel make ambiguous) or of the default
  start (which an angle offset would), they need not lie on the patch, and
  one sentence covers all shapes. The nearest candidate is taken at any
  distance (a far point is a legitimate way to say "the corner on that
  side") and logged; two candidates within 1% of the patch's size (the
  square root of its area) of each other are an error that lists them, as
  are a ring's or disk's faces on different sides (more than an eighth of a
  turn apart) and a `toward` within 1% of a turn of the origin or of half
  way around. `toward` is optional: without it the side
  `u` runs along by default is kept. `corners` remains for the other
  question, which nodes of an outline without four clear corners are the
  corners: four points, whose order then also fixes the start; `[]` makes a
  disk. The frame's three keys are gone.

## Shapes

The faces of the entry (after `where`), gathered from every rank, are
triangulated (quadrilaterals by the diagonal from their lowest node) and
their edges counted. Edges with one face form the outline. One connected
surface with one loop (Euler characteristic 1) is four-sided or a disk; with
two loops (characteristic 0) a ring. Separate pieces, closed surfaces, more
loops, handles, edges with more than two faces and pinched outlines are
`InputError`s that name the entry and say to split it with `where`.

A node of a single loop is a corner if the outline turns there by more than
30 degrees. Four corners make a four-sided patch, none a disk, any other
count an error that lists them and asks for `corners`. 30 degrees keeps the
obtuse corners of a parallelogram down to 150 degrees inside and lets a
circle of 13 or more faces pass as smooth.

## Coordinates

All of them come from the cotangent Laplacian `K` of the triangulated
surface (piecewise linear elements).

- **Four-sided.** `u` and `v` solve `K u = 0` with Dirichlet values on the
  whole outline: 0 and 1 on two opposite sides, and the fraction of arc
  length along the other two. Affine coordinates of a planar parallelogram
  satisfy exactly this problem and linear elements hold them, so they are
  reproduced to round-off (tested). Of the eight ways to label the sides,
  the one whose `u` and `v` directions (between the centers of opposite
  sides) are nearest to the axis after the patch's mean normal's dominant
  one, and the one after that, is taken (the sum of the two cosines; the
  first labeling of equal ones, in the order of the corners around the
  outline from its lowest node). `origin` keeps the labelings that start at
  its corner and `toward` the one of those whose `u` side ends nearest to
  it; with `origin` alone the absolute cosines decide, so `u` keeps its
  default pair of sides.
- **Ring.** `v` solves `K v = 0` with 0 on one rim and 1 on the other. Its
  harmonic conjugate is computed on the triangles (linear on each, continuous
  at edge midpoints, gradient that of `v` turned by 90 degrees in the
  surface); around the ring it gains the flux of `v`, `P = sum over a rim of
  (K v)_i`, so `u` is the conjugate over `P`, modulo 1. On an annulus this is
  `log(r / r0) / log(r1 / r0)` and the angle, on a cone the same in the
  distance from the apex, on a tube the height and the angle (tested against
  each). `v = 0` is the shorter rim, or the lower of two equal ones.
- **Disk.** A harmonic function with one value on the edge is constant, so
  the disk uses the Green's function of its center instead: `K G = delta`
  with `G = 0` on the edge and the unit source at the surface's point
  nearest to the centroid of its area, spread over the nodes of the triangle
  that holds it. `f = exp(-2 pi (G + i G*))`, with `G*` the conjugate of
  `G`, is the conformal map of the patch to the unit disk that sends the
  center to 0: `v = |f|` is the radius fraction (`r / R` on a circle), 1 on
  the edge of any outline, and `u = arg(f) / 2 pi` the angle. `G` and `G*`
  are singular at the center (a logarithm and an angle), and a first version
  that took `u` from the discrete conjugate there gave lines of constant `u`
  that were parallel within each triangle at the center instead of rays. `f`
  itself is smooth, and linear in the position near the center, so it is
  what is discretized: on the edge `|f| = 1` and its angle grows from node to
  node by the flux of the discrete `G` through the edge (the reactions `(K
  G)_i`, which sum to 1), and inside `Re f` and `Im f` solve Laplace's
  equation with those values. Face values are those of the linear `f` at the
  centroid. For a circle the edge values are those of `(x - center) / R`,
  which linear elements hold, so `v` and `u` are exact to round-off through
  the center too (tested).
- **Where `u` starts on rings and disks.** The angle is defined up to a
  constant and a direction. Both are set so that `u` is, on area-weighted
  average, the angle about the patch's center from a first direction toward
  a second, over `2 pi`. With `a` the ring's axis (the mean normal of a disk
  or a flat ring, the line between the rims' centers of a tube) and `k` the
  coordinate with the largest `|a_k|` (the lowest of equal ones), the first
  direction is the coordinate axis `k + 1` (cyclic) minus its part along
  `a`, which cannot vanish, and the second `a` x first, turned so that its
  largest component is positive. On surfaces of revolution `u` is then the
  geometric angle to round-off. `origin` replaces the constant (`u` minus its
  value at the face nearest to the point) and `toward` the direction (`u` is
  reversed if the face nearest to it is more than half way around).
- **Lengths.** Fast marching on the triangles from the nodes where `v` (or
  `u`) is 0, with the planar-front update where the front enters a triangle
  through its far edge and the edge lengths otherwise; first order, exact for
  the distance from a straight side.
- **Faces.** Each face takes the area-weighted mean over its triangles of the
  linear function at their centroids, which is the function at the face's
  centroid when it is linear.
- **2D.** The faces of an entry are chained into one open curve; `u_m` is the
  arc length to the face's center from the end with the lower coordinate
  along the dominant axis of the line between the ends, `u` that over the
  curve's length.

## Parallel runs and cost

Every rank gathers the entry's faces (nodes and inward area vectors, sorted
by a key every partition agrees on: the owning cell's global index and the
face's position in it), builds the same triangulation with nodes ordered by
their coordinates, and runs the same serial solves, so each face's values
are the same bits on any rank count (tested on 1 and 3 ranks). Only entries
whose expressions read `u`, `v` or their other names pay for it. The solves
are Jacobi-preconditioned conjugate gradients to a relative residual of
1e-14, once at setup: two for a four-sided patch, one for a ring, three for
a disk.
The iteration count grows with the number of faces across the patch, which
is acceptable for inlets (10^5 faces in seconds); a multigrid preconditioner
is the next step if patches of 10^7 faces appear.

## Limits

- Patches with more than one hole, or with handles, have no single `u`, `v`
  here; they are errors with the fix (split with `where`).
- A patch with corners other than four needs `corners`.
- On meshes with strongly obtuse boundary triangles the cotangent weights
  can be negative and `u`, `v` can leave [0, 1] by a little near the outline;
  face values are clamped to [0, 1].
- The coordinates are those of the mesh the run starts with.
