%% User set parameters

%the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rx, ry, rz)
tform = [0, 0, 0, -pi/2, 0, 3];

%camera intrinsic parameters
%can be either 3x4 camera matrix or [f, cx, cy] or [fx, fy, cx, fy]
cam = [760, size(base{1}.v,2)/2,size(base{1}.v,1)/2];

%% Setup

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'None';

%get scans and images
move = getPointClouds(1);
move{1} = move{1}(:,:);
base = getImagesC(1, false);

Setup(metric, move, base, tform, cam, true);

%% Colour and output scan
scan = ColourScan(0);

dlmwrite('ScanOut.csv',scan,'precision',12 );

%% Clean up
ClearEverything();