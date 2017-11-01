sizeo = .3;
a = 2*Pi;
Point(1) = {0, 0, 0, sizeo};
Point(2) = {a, 0, 0, sizeo};
Point(3) = {a, a, 0, sizeo};
Point(4) = {0, a, 0, sizeo};

r = 0.5;
sizei = 0.1;
Point(5) = {a/2, a/2, 0, 1};
Point(6) = {a/2+r, a/2, 0, sizei};
Point(7) = {a/2-r, a/2, 0, sizei};
Point(8) = {a/2, a/2+r, 0, sizei};
Point(9) = {a/2, a/2-r, 0, sizei};

Point(10) = {a/2, a/2+3*r, 0, sizei};
Point(11) = {a/2, a/2+2*r, 0, sizei};
Point(12) = {a/2+r, a/2+3*r, 0, sizei};
Point(13) = {a/2, a/2+4*r, 0, sizei};
Point(14) = {a/2-r, a/2+3*r, 0, sizei};

Point(15) = {a/2, a/2-3*r, 0, sizei};
Point(16) = {a/2, a/2-2*r, 0, sizei};
Point(17) = {a/2+r, a/2-3*r, 0, sizei};
Point(18) = {a/2, a/2-4*r, 0, sizei};
Point(19) = {a/2-r, a/2-3*r, 0, sizei};

Point(20) = {a/2+3*r, a/2, 0, sizei};
Point(21) = {a/2+4*r, a/2, 0, sizei};
Point(22) = {a/2+3*r, a/2+r, 0, sizei};
Point(23) = {a/2+2*r, a/2, 0, sizei};
Point(24) = {a/2+3*r, a/2-r, 0, sizei};

Line(1) = {4, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Line(4) = {1, 4};

Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};

Circle(12) = {11, 10, 12};
Circle(13) = {12, 10, 13};
Circle(14) = {13, 10, 14};
Circle(15) = {14, 10, 11};

Circle(16) = {18, 15, 17};
Circle(17) = {17, 15, 16};
Circle(18) = {16, 15, 19};
Circle(19) = {19, 15, 18};

Circle(20) = {24, 20, 21};
Circle(21) = {21, 20, 22};
Circle(22) = {22, 20, 23};
Circle(23) = {23, 20, 24};

Line Loop(9) = {4, 1, 2, 3};
Line Loop(10) = {5, 6, 7, 8};
Line Loop(11) = {12, 13, 14, 15};
Line Loop(12) = {16, 17, 18, 19};
Line Loop(13) = {20, 21, 22, 23};

Plane Surface(11) = {9, 10, 11, 12, 13};


//Physical Line(11) = {5, 6, 7, 8};
Physical Surface(1) = {11};

Periodic Line {1} = {-3};
Periodic Line {2} = {-4};


