sizeo = .4;
a = 2*Pi;
Point(1) = {0, 0, 0, sizeo};
Point(2) = {a, 0, 0, sizeo};
Point(3) = {a, a, 0, sizeo};
Point(4) = {0, a, 0, sizeo};
Point(5) = {0, 0, a, sizeo};
Point(6) = {a, 0, a, sizeo};
Point(7) = {a, a, a, sizeo};
Point(8) = {0, a, a, sizeo};

r = 0.5;
h = 2.0;
sizei = .1;
Point(9) = {a/2, a/2, a/2-h/2, 1};
Point(10) = {a/2+r, a/2, a/2-h/2, sizei};
Point(11) = {a/2-r, a/2, a/2-h/2, sizei};
Point(12) = {a/2, a/2+r, a/2-h/2, sizei};
Point(13) = {a/2, a/2-r, a/2-h/2, sizei};
Point(14) = {a/2, a/2, a/2+h/2, 1};
Point(15) = {a/2+r, a/2, a/2+h/2, sizei};
Point(16) = {a/2-r, a/2, a/2+h/2, sizei};
Point(17) = {a/2, a/2+r, a/2+h/2, sizei};
Point(18) = {a/2, a/2-r, a/2+h/2, sizei};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {1, 5};
Line(6) = {5, 6};
Line(7) = {6, 2};
Line(8) = {6, 7};
Line(9) = {7, 3};
Line(10) = {7, 8};
Line(11) = {8, 4};
Line(12) = {8, 5};

Circle(13) = {16, 14, 17};
Circle(14) = {17, 14, 15};
Circle(15) = {15, 14, 18};
Circle(16) = {18, 14, 16};
Circle(17) = {12, 9, 10};
Circle(18) = {10, 9, 13};
Circle(19) = {13, 9, 11};
Circle(20) = {11, 9, 12};
Line(21) = {12, 17};
Line(22) = {10, 15};
Line(23) = {13, 18};
Line(24) = {11, 16};

Line Loop(25) = {15, 16, 13, 14};
Plane Surface(26) = {25};
Line Loop(27) = {20, 17, 18, 19};
Plane Surface(28) = {27};
Line Loop(29) = {20, 21, -13, -24};
Ruled Surface(30) = {29};
Line Loop(31) = {19, 24, -16, -23};
Ruled Surface(32) = {31};
Line Loop(33) = {18, 23, -15, -22};
Ruled Surface(34) = {33};
Line Loop(35) = {17, 22, -14, -21};
Ruled Surface(36) = {35};

Line Loop(37) = {4, 1, 2, 3};
Plane Surface(38) = {37};
Line Loop(39) = {5, -12, 11, 4};
Plane Surface(40) = {39};
Line Loop(41) = {11, -3, -9, 10};
Plane Surface(42) = {41};
Line Loop(43) = {-7, 8, 9, -2};
Plane Surface(44) = {43};
Line Loop(45) = {12, 6, 8, 10};
Plane Surface(46) = {45};
Line Loop(47) = {-5, 1, -7, -6};
Plane Surface(48) = {47};


Surface Loop(49) = {42, 44, 48, 40, 38, 46};

Surface Loop(50) = {26, 32, 30, 36, 34, 28};

Volume(51) = {49, 50};
Physical Volume(1) = {51};

Periodic Surface 38 {4, 1, 2, 3} = 46 {12, 6, 8, 10};
Periodic Surface 40 {5, -12, 11, 4} = 44 {-7, 8, 9, -2};
Periodic Surface 42 {11, -3, -9, 10} = 48 {-5, 1, -7, -6};
