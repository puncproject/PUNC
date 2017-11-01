rp = 0.000145;
ro = 10.*rp;
reso = ro/5.;
resp = rp/5.;

Point(1) = {ro, ro, 0, resp};

// Interior Boundary
Point(2) = {ro+rp, ro   , 0, resp};
Point(3) = {ro-rp, ro   , 0, resp};
Point(4) = {ro   , ro+rp, 0, resp};
Point(5) = {ro   , ro-rp, 0, resp};
Circle(1) = {4, 1, 3};
Circle(2) = {3, 1, 5};
Circle(3) = {5, 1, 2};
Circle(4) = {2, 1, 4};

// Exterior Boundary
Point(6) = {   0,    0, 0, reso};
Point(7) = {   0, 2*ro, 0, reso};
Point(8) = {2*ro, 2*ro, 0, reso};
Point(9) = {2*ro,    0, 0, reso};
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 9};
Line(8) = {9, 6};

Physical Line(9) = {5, 6, 7, 8};  // exterior boundary
Physical Line(10) = {1, 2, 3, 4}; // interior boundary
Line Loop(11) = {5, 6, 7, 8};
Line Loop(12) = {1, 2, 3, 4};
Plane Surface(13) = {11, 12};
Physical Surface(14) = {13};
