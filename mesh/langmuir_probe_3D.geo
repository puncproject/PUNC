// boom parameters
xb0 = 0.;
yb0 = 0.;
zb0 = 0.;
rb = 0.0011;
lb = 0.133/10.;
resb = rb/5.;

// guard cylinder parameters
rg = 0.000595;
lg = 0.015/10.;
resg = rg/5.;

// needle probe parameters
rp = 0.000145;
lp = 0.04/10.;
resp = rp/5.;

// boom geometry
Point(1) = {xb0,    yb0,    zb0,    resb};
Point(2) = {xb0+rb, yb0,    zb0,    resb};
Point(3) = {xb0-rb, yb0,    zb0,    resb};
Point(4) = {xb0,    yb0+rb, zb0,    resb};
Point(5) = {xb0,    yb0-rb, zb0,    resb};
Point(6) = {xb0,    yb0,    zb0+lb, resb};
Point(7) = {xb0+rb, yb0,    zb0+lb, resb};
Point(8) = {xb0-rb, yb0,    zb0+lb, resb};
Point(9) = {xb0,    yb0+rb, zb0+lb, resb};
Point(10)= {xb0,    yb0-rb, zb0+lb, resb};
Circle(1) = {4, 1, 2};
Circle(2) = {2, 1, 5};
Circle(3) = {5, 1, 3};
Circle(4) = {3, 1, 4};
Circle(5) = {7, 6, 9};
Circle(6) = {9, 6, 8};
Circle(7) = {8, 6, 10};
Circle(8) = {10, 6, 7};
Line(9) = {2, 7};
Line(10) = {4, 9};
Line(11) = {3, 8};
Line(12) = {5, 10};

// gard geometry
Point(20) = {xb0,    yb0,    zb0+lb,    resg};
Point(21) = {xb0+rg, yb0,    zb0+lb,    resg};
Point(22) = {xb0-rg, yb0,    zb0+lb,    resg};
Point(23) = {xb0,    yb0+rg, zb0+lb,    resg};
Point(24) = {xb0,    yb0-rg, zb0+lb,    resg};
Point(25) = {xb0,    yb0,    zb0+lb+lg, resg};
Point(26) = {xb0+rg, yb0,    zb0+lb+lg, resg};
Point(27) = {xb0-rg, yb0,    zb0+lb+lg, resg};
Point(28) = {xb0,    yb0+rg, zb0+lb+lg, resg};
Point(29)= {xb0,     yb0-rg, zb0+lb+lg, resg};
Circle(13) = {23, 6, 22};
Circle(14) = {22, 6, 24};
Circle(15) = {24, 6, 21};
Circle(16) = {21, 6, 23};
Circle(17) = {28, 25, 27};
Circle(18) = {27, 25, 29};
Circle(19) = {29, 25, 26};
Circle(20) = {26, 25, 28};
Line(21) = {23, 28};
Line(22) = {22, 27};
Line(23) = {24, 29};
Line(24) = {21, 26};

// deedle probe
Point(30) = {xb0,    yb0,    zb0+lb+lg,    resg};
Point(31) = {xb0+rp, yb0,    zb0+lb+lg,    resp};
Point(32) = {xb0-rp, yb0,    zb0+lb+lg,    resp};
Point(33) = {xb0,    yb0+rp, zb0+lb+lg,    resp};
Point(34) = {xb0,    yb0-rp, zb0+lb+lg,    resp};
Point(35) = {xb0,    yb0,    zb0+lb+lg+lp, resg};
Point(36) = {xb0+rp, yb0,    zb0+lb+lg+lp, resp};
Point(37) = {xb0-rp, yb0,    zb0+lb+lg+lp, resp};
Point(38) = {xb0,    yb0+rp, zb0+lb+lg+lp, resp};
Point(39) = {xb0,    yb0-rp, zb0+lb+lg+lp, resp};
Circle(25) = {33, 25, 32};
Circle(26) = {32, 25, 34};
Circle(27) = {34, 25, 31};
Circle(28) = {31, 25, 33};
Circle(29) = {38, 35, 37};
Circle(30) = {37, 35, 39};
Circle(31) = {39, 35, 36};
Circle(32) = {36, 35, 38};
Line(33) = {33, 38};
Line(34) = {32, 37};
Line(35) = {34, 39};
Line(36) = {31, 36};
Line Loop(37) = {32, 29, 30, 31};
Plane Surface(38) = {37};
Line Loop(39) = {20, 17, 18, 19};
Line Loop(40) = {28, 25, 26, 27};
Plane Surface(41) = {39, 40};
Line Loop(42) = {5, 6, 7, 8};
Line Loop(43) = {15, 16, 13, 14};
Plane Surface(44) = {42, 43};
Line Loop(45) = {4, 1, 2, 3};
Plane Surface(46) = {45};
Line Loop(47) = {31, -36, -27, 35};
Ruled Surface(48) = {47};
Line Loop(49) = {32, -33, -28, 36};
Ruled Surface(50) = {49};
Line Loop(51) = {33, 29, -34, -25};
Ruled Surface(52) = {51};
Line Loop(53) = {34, 30, -35, -26};
Ruled Surface(54) = {53};
Line Loop(55) = {20, -21, -16, 24};
Ruled Surface(56) = {55};
Line Loop(57) = {21, 17, -22, -13};
Ruled Surface(58) = {57};
Line Loop(59) = {18, -23, -14, 22};
Ruled Surface(60) = {59};
Line Loop(61) = {19, -24, -15, 23};
Ruled Surface(62) = {61};
Line Loop(63) = {5, -10, 1, 9};
Ruled Surface(64) = {63};
Line Loop(65) = {8, -9, 2, 12};
Ruled Surface(66) = {65};
Line Loop(67) = {6, -11, 4, 10};
Ruled Surface(68) = {67};
Line Loop(69) = {11, 7, -12, 3};
Ruled Surface(70) = {69};

