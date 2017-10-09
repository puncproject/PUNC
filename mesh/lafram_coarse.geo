l = 10;			// length of cube
c = l/2;		// center of cube
r = 1;			// probe radius
reso = 2.5;		// outer resolution
resi = 0.5;		// inner resolution

// Outer boundary (box)
Point(1) = {0,0,0, reso};
Point(2) = {0,0,l, reso};
Point(3) = {0,l,0, reso};
Point(4) = {0,l,l, reso};
Point(5) = {l,0,0, reso};
Point(6) = {l,0,l, reso};
Point(7) = {l,l,0, reso};
Point(8) = {l,l,l, reso};

// Center
Point(9) = {c, c, c, resi};

// Inner boundary (probe sphere)
Point(10) = {c+r, c, c, resi};
Point(11) = {c, c+r, c, resi};
Point(12) = {c, c, c+r, resi};
Point(13) = {c-r, c, c, resi};
Point(14) = {c, c-r, c, resi};
Point(15) = {c, c, c-r, resi};

// Outer boundary
Line(1) = {2, 6};
Line(2) = {6, 8};
Line(3) = {8, 4};
Line(4) = {4, 2};
Line(5) = {2, 1};
Line(6) = {1, 3};
Line(7) = {3, 4};
Line(8) = {3, 7};
Line(9) = {7, 8};
Line(10) = {7, 5};
Line(11) = {5, 6};
Line(12) = {5, 1};

// Inner boundary
Circle(13) = {11, 9, 10};
Circle(14) = {10, 9, 14};
Circle(15) = {14, 9, 13};
Circle(16) = {13, 9, 11};
Circle(17) = {11, 9, 15};
Circle(18) = {15, 9, 14};
Circle(19) = {14, 9, 12};
Circle(20) = {12, 9, 11};
Circle(21) = {13, 9, 12};
Circle(22) = {12, 9, 10};
Circle(23) = {10, 9, 15};
Circle(24) = {15, 9, 13};

// Outer boundary
Line Loop(25) = {4, 1, 2, 3};
Plane Surface(26) = {25};
Line Loop(27) = {2, -9, 10, 11};
Plane Surface(28) = {27};
Line Loop(29) = {12, 6, 8, 10};
Plane Surface(30) = {29};
Line Loop(31) = {6, 7, 4, 5};
Plane Surface(32) = {31};
Line Loop(33) = {5, -12, 11, -1};
Plane Surface(34) = {33};
Line Loop(35) = {9, 3, -7, 8};
Plane Surface(36) = {35};

// Inner boundary
Line Loop(37) = {20, 13, -22};
Ruled Surface(38) = {37};
Line Loop(39) = {13, 23, -17};
Ruled Surface(40) = {39};
Line Loop(41) = {17, 24, 16};
Ruled Surface(42) = {41};
Line Loop(43) = {16, -20, -21};
Ruled Surface(44) = {43};
Line Loop(45) = {21, -19, 15};
Ruled Surface(46) = {45};
Line Loop(47) = {24, -15, -18};
Ruled Surface(48) = {47};
Line Loop(49) = {18, -14, 23};
Ruled Surface(50) = {49};
Line Loop(51) = {14, 19, 22};
Ruled Surface(52) = {51};

// Outer boundary
Physical Surface(53) = {36, 32, 28, 34, 30, 26};

// Inner boundary
Physical Surface(54) = {40, 38, 42, 44, 46, 48, 50, 52};

// Outer boundary
Surface Loop(55) = {32, 30, 34, 28, 26, 36};

// Inner boundary
Surface Loop(56) = {38, 44, 42, 40, 50, 48, 46, 52};

// Volume
Volume(57) = {55, 56};
Physical Volume(58) = {57};
