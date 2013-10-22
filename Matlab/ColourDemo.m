%% User set parameters

%the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rx, ry, rz)
tform = [-7.44701810563511,-0.232207616254354,3.86240686947799,-1.45778878054023,0.0263305926601604,0.963238545850066];

%camera intrinsic parameters
%can be either 3x4 camera matrix or [f, cx, cy] or [fx, fy, cx, fy]
cam = [5893.21071816293,6249.50000000000,800];

%% Setup

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'None';

%get scans and images
move = getPointClouds(1);
move{1} = move{1}(:,:)- repmat([10455.289,7824.418,681.748 0],size(move{1},1),1);
base = getImagesC(1, false);

Setup(metric, move, base, tform, cam, true);

%% Colour and output scan
scan = ColourScan(0);

scan = scan + repmat([-7.44701810563511,-0.232207616254354,3.86240686947799, 0,0,0],size(scan,1),1);

dlmwrite('ScanOut.csv',scan,'precision',12 );

%% Clean up
ClearEverything();