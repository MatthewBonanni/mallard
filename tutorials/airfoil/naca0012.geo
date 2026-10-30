// C-grid of quadrilaterals around a NACA 0012 airfoil of unit chord with a
// closed trailing edge (the High-Order CFD Workshop's definition: the x^4
// coefficient is -0.1036 instead of -0.1015).
//
//   gmsh naca0012.geo -2 -o naca0012.msh
//   gmsh naca0012.geo -2 -setnumber refine 2 -o naca0012.msh
//
// Six blocks: above and below the airfoil, bounded by a half circle of
// radius R about the trailing edge; above and below the near wake, from the
// trailing edge to x = 1 + W, where the wake is resolved; and above and
// below the far wake, to the outlet at x = 1 + R. Boundary zones: "airfoil"
// (the wall) and "farfield" (the half circle, the outlet and the lines that
// join them).

DefineConstant[
  refine = {1, Name "Refinement factor (multiplies every cell count)"},
  R = {100, Name "Far-field distance (chords)"},
  W = {4, Name "Length of the near wake (chords)"},
  n_surf = {96, Name "Cells along each surface"},
  n_normal = {64, Name "Cells from the wall to the far field"},
  n_near = {96, Name "Cells along the near wake"},
  n_far = {32, Name "Cells along the far wake"},
  dn_wall = {1e-3, Name "First cell height at the wall (chords)"},
  ds_wake = {3e-3, Name "First wake cell at the trailing edge (chords)"},
  bump = {0.12, Name "Bump factor of the surface spacing (smaller: finer at both edges)"},
  n_spline = {120, Name "Spline points per surface"}
];

ns = Round(n_surf * refine);
nn = Round(n_normal * refine);
nw1 = Round(n_near * refine);
nw2 = Round(n_far * refine);
dn = dn_wall / refine;
ds_te = ds_wake / refine;

LE = newp; Point(LE) = {0, 0, 0};
TE = newp; Point(TE) = {1, 0, 0};
upper[] = {LE};
lower[] = {LE};
For i In {1:n_spline - 1}
  x = 0.5 * (1 - Cos(Pi * i / n_spline));
  y = 0.6 * (0.2969 * Sqrt(x) - 0.1260 * x - 0.3516 * x^2 + 0.2843 * x^3 - 0.1036 * x^4);
  p = newp; Point(p) = {x, y, 0};
  upper[] += {p};
  p = newp; Point(p) = {x, -y, 0};
  lower[] += {p};
EndFor
upper[] += {TE};
lower[] += {TE};

top = newp; Point(top) = {1, R, 0};
front = newp; Point(front) = {1 - R, 0, 0};
bottom = newp; Point(bottom) = {1, -R, 0};
mid = newp; Point(mid) = {1 + W, 0, 0};
mid_top = newp; Point(mid_top) = {1 + W, R, 0};
mid_bottom = newp; Point(mid_bottom) = {1 + W, -R, 0};
wake = newp; Point(wake) = {1 + R, 0, 0};
wake_top = newp; Point(wake_top) = {1 + R, R, 0};
wake_bottom = newp; Point(wake_bottom) = {1 + R, -R, 0};

s_upper = newc; Spline(s_upper) = upper[];
s_lower = newc; Spline(s_lower) = lower[];
te_top = newc; Line(te_top) = {TE, top};
arc_top = newc; Circle(arc_top) = {top, TE, front};
le_front = newc; Line(le_front) = {LE, front};
te_bottom = newc; Line(te_bottom) = {TE, bottom};
arc_bottom = newc; Circle(arc_bottom) = {bottom, TE, front};
cut_near = newc; Line(cut_near) = {TE, mid};
cut_far = newc; Line(cut_far) = {mid, wake};
mid_up = newc; Line(mid_up) = {mid, mid_top};
mid_down = newc; Line(mid_down) = {mid, mid_bottom};
out_top = newc; Line(out_top) = {wake, wake_top};
out_bottom = newc; Line(out_bottom) = {wake, wake_bottom};
near_top = newc; Line(near_top) = {top, mid_top};
near_bottom = newc; Line(near_bottom) = {bottom, mid_bottom};
far_top = newc; Line(far_top) = {mid_top, wake_top};
far_bottom = newc; Line(far_bottom) = {mid_bottom, wake_bottom};

// Lines away from the wall: first cell dn, growing geometrically by q with
// dn (q^n - 1) / (q - 1) = length (solved by fixed point)
q_te = 1.1; q_le = 1.1;
For it In {1:200}
  q_te = (1 + R * (q_te - 1) / dn)^(1 / nn);
  q_le = (1 + (R - 1) * (q_le - 1) / dn)^(1 / nn);
EndFor
Transfinite Curve{te_top, te_bottom, mid_up, mid_down, out_top, out_bottom} = nn + 1 Using Progression q_te;
Transfinite Curve{le_front} = nn + 1 Using Progression q_le;
// Surfaces, clustered at both edges, and the arcs opposite them
Transfinite Curve{s_upper, s_lower} = ns + 1 Using Bump bump;
Transfinite Curve{arc_top, arc_bottom} = ns + 1;
// Near wake: from the trailing-edge spacing over W; far wake: from the last
// near-wake cell to the outlet
q_1 = 1.05; q_2 = 1.1;
For it In {1:200}
  q_1 = (1 + W * (q_1 - 1) / ds_te)^(1 / nw1);
EndFor
ds_mid = ds_te * q_1^(nw1 - 1);
For it In {1:200}
  q_2 = (1 + (R - W) * (q_2 - 1) / ds_mid)^(1 / nw2);
EndFor
Transfinite Curve{cut_near, near_top, near_bottom} = nw1 + 1 Using Progression q_1;
Transfinite Curve{cut_far, far_top, far_bottom} = nw2 + 1 Using Progression q_2;

loops[] = {};
cl = newcl; Curve Loop(cl) = {s_upper, te_top, arc_top, -le_front}; loops[] += {cl};
cl = newcl; Curve Loop(cl) = {-s_lower, le_front, -arc_bottom, -te_bottom}; loops[] += {cl};
cl = newcl; Curve Loop(cl) = {cut_near, mid_up, -near_top, -te_top}; loops[] += {cl};
cl = newcl; Curve Loop(cl) = {-cut_near, te_bottom, near_bottom, -mid_down}; loops[] += {cl};
cl = newcl; Curve Loop(cl) = {cut_far, out_top, -far_top, -mid_up}; loops[] += {cl};
cl = newcl; Curve Loop(cl) = {-cut_far, mid_down, far_bottom, -out_bottom}; loops[] += {cl};
blocks[] = {};
For b In {0:5}
  s = news; Plane Surface(s) = {loops[b]};
  Transfinite Surface{s};
  Recombine Surface{s};
  blocks[] += {s};
EndFor

Physical Curve("airfoil") = {s_upper, s_lower};
Physical Curve("farfield") = {arc_top, arc_bottom, out_top, out_bottom, near_top, near_bottom, far_top, far_bottom};
Physical Surface("fluid") = {blocks[]};
