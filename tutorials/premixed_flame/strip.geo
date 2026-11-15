// A strip of n quadrilaterals for a one-dimensional flame: length L along x,
// one cell across (cells ten times longer across than along, so the faces
// across the strip do not limit the time step).
//
//   gmsh strip.geo -2 -o strip.msh                       (the phi = 1 flame)
//   gmsh strip.geo -2 -setnumber L 0.006 -setnumber n 300 -o strip.msh
//
// Boundary zones: "inlet" (x = 0), "outlet" (x = L) and "sides".

DefineConstant[
  L = {4.9366e-3, Name "Length (m)"},
  n = {300, Name "Cells along x"}
];
h = 10 * L / n;
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, h, 0};
Point(4) = {0, h, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Curve{1, 3} = n + 1;
Transfinite Curve{2, 4} = 2;
Transfinite Surface{1};
Recombine Surface{1};
Physical Curve("inlet") = {4};
Physical Curve("outlet") = {2};
Physical Curve("sides") = {1, 3};
Physical Surface("gas") = {1};
