// Gmsh project created on Tue Jan 10 16:40:57 2017
//+
Point(1) = {0, 0, 0, 0.1};
//+
Point(2) = {1, 0, 0, 0.1};
//+
Point(3) = {1, 1, 0, 0.1};
//+
Point(4) = {0, 1, 0, 0.1};
//+
Point(5) = {0.5, 0.5, 0, 0.1};
//+
Line(1) = {4, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 1};
//+
Line(4) = {1, 4};
//+
Point(6) = {0.5, 0.6, 0, 0.1};
//+
Circle(5) = {6, 5, 6};
//+
Line Loop(6) = {4, 1, 2, 3};
//+
Line Loop(7) = {5};
//+
Ruled Surface(8) = {6, 7};
//+
Delete {
  Surface{8};
}
//+
Plane Surface(8) = {6, 7};
//+
Characteristic Length {6} = 0.02;
