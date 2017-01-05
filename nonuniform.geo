// Gmsh project created on Thu Dec 22 09:07:50 2016

Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 3.14, 0, 1.0};
Point(3) = {0, 6.28, 0, 1.0};
Point(4) = {3.14, 6.28, 0, 1.0};
Point(5) = {3.14, 3.14, 0, 1.0};
Point(6) = {3.14, 0, 0, 1.0};
Point(7) = {6.28, 0, 0, 1.0};
Point(8) = {6.28, 3.14, 0, 1.0};
Point(9) = {6.28, 6.28, 0, 1.0};

Characteristic Length {5} = 0.1;

Line(1) = {1, 6};
Line(2) = {6, 5};
Line(3) = {5, 2};
Line(4) = {2, 1};
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 5};
Line(8) = {8, 9};
Line(9) = {9, 4};
Line(10) = {4, 5};
Line(11) = {4, 3};
Line(12) = {3, 2};

Line Loop(13) = {11, 12, -3, -10};
Plane Surface(14) = {13};
Line Loop(15) = {10, -7, 8, 9};
Plane Surface(16) = {15};
Line Loop(17) = {7, -2, 5, 6};
Plane Surface(18) = {17};
Line Loop(19) = {2, 3, 4, 1};
Plane Surface(20) = {19};

//Physical Line(21) = {4, 12};
//Physical Line(22) = {6, 8};
//Physical Line(23) = {5, 1};
//Physical Line(24) = {11, 9};

Periodic Line {4,12} = {6,8};
Periodic Line {1,5} = {11,9};
