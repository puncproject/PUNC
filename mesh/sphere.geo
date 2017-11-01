sizeo = 0.8;
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
sizei = 0.3;
Point(9) = {a/2, a/2, a/2, 1};
Point(10) = {a/2+r, a/2, a/2, sizei};
Point(11) = {a/2-r, a/2, a/2, sizei};
Point(12) = {a/2, a/2+r, a/2, sizei};
Point(13) = {a/2, a/2-r, a/2, sizei};
Point(14) = {a/2, a/2, a/2+r, sizei};
Point(15) = {a/2, a/2, a/2-r, sizei};


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

Circle(13) = {10, 9, 15};
Circle(14) = {14, 9, 10};
Circle(15) = {14, 9, 11};
Circle(16) = {11, 9, 15};
Circle(17) = {14, 9, 12};
Circle(18) = {12, 9, 15};
Circle(19) = {14, 9, 13};
Circle(20) = {13, 9, 15};


Line Loop(21) = {9, -2, -7, 8};
Plane Surface(22) = {21};
Line Loop(23) = {11, 4, 5, -12};
Plane Surface(24) = {23};
Line Loop(25) = {6, 8, 10, 12};
Plane Surface(26) = {25};
Line Loop(27) = {10, 11, -3, -9};
Plane Surface(28) = {27};
Line Loop(29) = {-6, -5, 1, -7};
Plane Surface(30) = {29};
Line Loop(31) = {1, 2, 3, 4};
Plane Surface(32) = {31};


Line Loop(35) = {17, 18, -13, -14};
Ruled Surface(35) = {35};
Line Loop(36) = {19, 20, -13, -14};
Ruled Surface(36) = {36};
Line Loop(37) = {15, 16, -18, -17};
Ruled Surface(37) = {37};
Line Loop(38) = {16, -20, -19, 15};
Ruled Surface(38) = {38};


Surface Loop(41) = {26, 30, 24, 28, 32, 22};

Surface Loop(42) = {36, 38, 37, 35};

Volume(43) = {41, 42};
Physical Volume(1) = {43};

Periodic Surface 22 {9, -2, -7, 8} = 24 {11, 4, 5, -12};
Periodic Surface 26 {6, 8, 10, 12} = 32 {1, 2, 3, 4};
Periodic Surface 28 {10, 11, -3, -9} = 30 {-6, -5, 1, -7};
