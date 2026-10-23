// Flow past a circular cylinder (cylinder tutorial, docs/tutorials/cylinder.md):
// a disk of radius R_far around a cylinder of diameter 1 at the origin.
// A ring of quadrilaterals hugs the wall (a structured O-grid, cells growing
// geometrically from dn0); triangles fill the rest, fine through the wake and
// coarsening toward the far field.
//
//   gmsh cylinder.geo -2 -o cylinder.msh [-setnumber h_wake 0.1] ...
//
// Physical groups: the curves "cylinder" and "farfield" (Mallard's boundary
// zones) and the surface "fluid".

DefineConstant[
  R_far = {20, Name "Far-field radius"},
  R_ring = {1.0, Name "Outer radius of the quadrilateral ring"},
  n_theta = {96, Name "Cells around the cylinder"},
  n_ring = {16, Name "Cells across the ring"},
  dn0 = {0.008, Name "First cell thickness at the wall"},
  h_wake = {0.12, Name "Cell size in the near wake"},
  L_wake = {12, Name "Length of the refined wake"},
  h_far = {2.0, Name "Cell size at the far field"}
];

// Growth ratio across the ring that gives a first cell of dn0
r = 1.1;
For k In {1:60}
  r = (1 + (R_ring - 0.5) * (r - 1) / dn0)^(1 / n_ring);
EndFor

Point(1) = {0, 0, 0};
For k In {0:3}
  a = k * Pi / 2 + Pi / 4;
  Point(10 + k) = {0.5 * Cos(a), 0.5 * Sin(a), 0};
  Point(20 + k) = {R_ring * Cos(a), R_ring * Sin(a), 0};
  Point(30 + k) = {R_far * Cos(a), R_far * Sin(a), 0};
EndFor
For k In {0:3}
  Circle(10 + k) = {10 + k, 1, 10 + (k + 1) % 4};
  Circle(20 + k) = {20 + k, 1, 20 + (k + 1) % 4};
  Circle(30 + k) = {30 + k, 1, 30 + (k + 1) % 4};
  Line(40 + k) = {10 + k, 20 + k};
EndFor

// The ring: four structured blocks of quadrilaterals
For k In {0:3}
  Curve Loop(50 + k) = {40 + k, 20 + k, -(40 + (k + 1) % 4), -(10 + k)};
  Plane Surface(50 + k) = {50 + k};
  Transfinite Curve{10 + k, 20 + k} = n_theta / 4 + 1;
  Transfinite Curve{40 + k} = n_ring + 1 Using Progression r;
  Transfinite Surface{50 + k};
  Recombine Surface{50 + k};
EndFor

// The rest: triangles
Curve Loop(60) = {30, 31, 32, 33};
Curve Loop(61) = {20, 21, 22, 23};
Plane Surface(60) = {60, 61};

Physical Curve("cylinder") = {10, 11, 12, 13};
Physical Curve("farfield") = {30, 31, 32, 33};
Physical Surface("fluid") = {50, 51, 52, 53, 60};

// Cell size of the triangles: that of the ring's outer cells at the ring,
// h_wake through the wake, growing to h_far at the far field
Field[1] = Distance;
Field[1].CurvesList = {20, 21, 22, 23};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 2 * Pi * R_ring / n_theta;
Field[2].SizeMax = h_far;
Field[2].DistMin = 0;
Field[2].DistMax = R_far - R_ring;
Field[3] = Box;
Field[3].VIn = h_wake;
Field[3].VOut = h_far;
Field[3].XMin = 0;
Field[3].XMax = L_wake;
Field[3].YMin = -1.5;
Field[3].YMax = 1.5;
Field[3].Thickness = 4;
Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.MshFileVersion = 2.2;
