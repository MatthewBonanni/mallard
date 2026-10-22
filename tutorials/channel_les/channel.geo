// Hexahedra for a plane channel between walls at y = 0 and y = 2h (h = 1),
// periodic in x and z: uniform in x and z, clustered toward both walls in y
// with the tanh law of Mallard's generated meshes, node j of ny at
// y_j = 1 + tanh(beta (2 j / ny - 1)) / tanh(beta).
//
//   gmsh channel.geo -3 -o channel.msh
//   gmsh channel.geo -3 -setnumber nx 64 -setnumber nz 64 -o channel_fine.msh
//
// The mesh is extruded from a line of nodes along y, first along x, then
// along z, so opposite faces carry the same nodes translated, which
// Mallard's [[periodic]] needs. Boundary zones: "bottom" (y = 0), "top"
// (y = 2), "left" (x = 0), "right" (x = Lx), "back" (z = 0), "front" (z = Lz).

DefineConstant[
  Lx = {2 * Pi, Name "Length (streamwise)"},
  Lz = {Pi, Name "Width (spanwise)"},
  nx = {32, Name "Cells along x"},
  ny = {48, Name "Cells along y"},
  nz = {32, Name "Cells along z"},
  beta = {2.0, Name "Wall clustering (0: uniform)"}
];

layers[] = {};
heights[] = {};
For j In {1:ny}
  layers[] += {1};
  heights[] += {0.5 * (1 + Tanh(beta * (2 * j / ny - 1)) / Tanh(beta))};
EndFor

Point(1) = {0, 0, 0};
line[] = Extrude {0, 2, 0} { Point{1}; Layers{layers[], heights[]}; };
plane[] = Extrude {Lx, 0, 0} { Curve{line[1]}; Layers{nx}; Recombine; };
box[] = Extrude {0, 0, Lz} { Surface{plane[1]}; Layers{nz}; Recombine; };

e = 1e-6;
Physical Surface("bottom") = Surface In BoundingBox{-e, -e, -e, Lx + e, e, Lz + e};
Physical Surface("top") = Surface In BoundingBox{-e, 2 - e, -e, Lx + e, 2 + e, Lz + e};
Physical Surface("left") = Surface In BoundingBox{-e, -e, -e, e, 2 + e, Lz + e};
Physical Surface("right") = Surface In BoundingBox{Lx - e, -e, -e, Lx + e, 2 + e, Lz + e};
Physical Surface("back") = Surface In BoundingBox{-e, -e, -e, Lx + e, 2 + e, e};
Physical Surface("front") = Surface In BoundingBox{-e, -e, Lz - e, Lx + e, 2 + e, Lz + e};
Physical Volume("fluid") = {box[1]};
