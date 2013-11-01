%% User set parameters

%the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rz, ry, rx)
tform = [-0.410703054243638,-0.0407604590635398,0.261366840786790,-1.55122664469742,-0.175985799310289,3.18818399174407];

%camera intrinsic parameters
%can be either 3x4 camera matrix or [f, cx, cy] or [fx, fy, cx, fy]
cam = [771.474058577767,1103.50000000000,160];

%% Setup

%metric to use
metric = 'None';

%get scans and images
move = GetPointClouds(1);
base = GetImagesC(1, false);

Setup(1,metric, move, base, tform, cam, true);

%% Colour and output scan
scan = ColourScan(0);

dlmwrite('ScanOut.csv',scan,'precision',12 );

%% Clean up
ClearEverything();