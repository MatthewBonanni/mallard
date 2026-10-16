// Quarter domain around a sharp cone at zero incidence: tetrahedra, fine
// between the cone and its conical shock, coarse outside.
//
//   gmsh cone.geo -3 -o cone.msh
//   gmsh cone.geo -3 -setnumber h 0.01 -o cone_fine.msh
//
// The cone has its tip at the origin, its axis along x, half-angle
// cone_angle and length 1; the domain is the part of the cylinder of radius
// R about the x axis between x = -x_up and x = 1 with y >= 0 and z >= 0.
// Boundary zones: "cone", "inlet" (x = -x_up), "outlet" (x = 1), "outer"
// (the cylinder), "symmetry_y" (y = 0) and "symmetry_z" (z = 0).

SetFactory("OpenCASCADE");
DefineConstant[
  cone_angle = {15, Name "Cone half-angle (degrees)"},
  x_up = {0.2, Name "Domain length upstream of the tip"},
  R = {0.8, Name "Domain radius"},
  h = {0.02, Name "Cell size between the cone and the shock"},
  h_out = {0.08, Name "Cell size outside the shock layer"},
  layer_angle = {40, Name "Fine region: within this angle (degrees) of the axis, downstream of the tip"}
];

r_base = Tan(cone_angle * Pi / 180);
Cylinder(1) = {-x_up, 0, 0, 1 + x_up, 0, 0, R};
Cone(2) = {0, 0, 0, 1, 0, 0, 0, r_base};
Box(3) = {-x_up, 0, 0, 1 + x_up, R, R};
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
BooleanIntersection(5) = { Volume{4}; Delete; }{ Volume{3}; Delete; };

e = 1e-6;
inlet() = Surface In BoundingBox{-x_up - e, -e, -e, -x_up + e, R + e, R + e};
outlet() = Surface In BoundingBox{1 - e, -e, -e, 1 + e, R + e, R + e};
sym_y() = Surface In BoundingBox{-x_up - e, -e, -e, 1 + e, e, R + e};
sym_z() = Surface In BoundingBox{-x_up - e, -e, -e, 1 + e, R + e, e};
cone() = Surface In BoundingBox{-e, -e, -e, 1 + e, r_base + e, r_base + e};
cone() -= {outlet(), sym_y(), sym_z()};
all() = Boundary{ Volume{5}; };
outer() = all();
outer() -= {inlet(), outlet(), sym_y(), sym_z(), cone()};

Physical Surface("cone") = {cone()};
Physical Surface("inlet") = {inlet()};
Physical Surface("outlet") = {outlet()};
Physical Surface("outer") = {outer()};
Physical Surface("symmetry_y") = {sym_y()};
Physical Surface("symmetry_z") = {sym_z()};
Physical Volume("fluid") = {5};

// h inside the cone of half-angle layer_angle around the axis (which holds
// the shock), blending to h_out over a few cells outside it
t = Tan(layer_angle * Pi / 180);
Field[1] = MathEval;
Field[1].F = Sprintf("%g + (%g - %g) * Min(1, Max(0, (Sqrt(y^2 + z^2) - %g * Max(x, 0) - 0.03) / 0.15))", h, h_out, h, t);
Background Field = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm3D = 10;
