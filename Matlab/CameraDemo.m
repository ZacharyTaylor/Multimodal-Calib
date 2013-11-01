%Demo of the program aligning 1 scan with 1 image

%% User set parameters

%metric to use, can be one of the following
% SSD - normalized SSD metric
% MI - mutual information (uses a histogram method with 50 bins)
% NMI - normalized mutual information (uses same method as above)
% GOM - gradient orientation measure
% GOMS - modified GOM method that evalutes statistical significants of
% results. Usually outperforms GOM
% LEV - levinson's method (may have a bug, still working on)
% None - no metric assigned used in generating images and coloured scans
% Note (NMI, MI, LEV and GOM) multiplied by -1 to give minimums in
% optimization
metric = 'GOM';

%inital guess as to the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rz, ry, rx)
tform = [0, 0, 0, -pi/2, 0, pi];

%camera intrinsic parameters
%can be either 3x4 camera matrix or [f, cx, cy] or [fx, fy, cx, fy]
cam = [760, 1103.5, 160];

%range around transform where the true solution lies 
%can be the following forms
%[x,y,z]
%[x,y,z,rx,ry,rz]
%[x,y,z,rx,ry,rz,f,cx,cy] if camera of form [f,cx,cy]
%[x,y,z,rx,ry,rz,fx,fy,cx,cy] if camera of form [fx,fy,cx,cy]
%rotations in radians, rotation order rx, ry, rz)
range = [0.5 0.5 0.5 0.1 0.3 0.1 40 0 0];

%feature to use as intensity information of lidar scans. Options are 
%intensity - basic lidar intensity
%range - distance of points from the lidar
%normals - angle between line from lidar to point and a horizontal plane
feature = 'intensity';

%True for panoramic camera, false otherwise
panFlag = true;

%Sets the update rate in seconds of the output that can be used to evaluate
%the metrics progress. Updating involves transfering the whole image off
%the gpu and so for large scans causes a significant slow down (increase
%value to reduce this issue)
updatePeriod = 5;

%How much to dialate each point by when generating an image from it (only
%effects view generated in updates, does not change metric values)
dilate = 2;

%% Setup

%get scans and images
move = GetPointClouds(1);
base = GetImagesC(1, false);

%get features for scans
for i = 1:size(move,1)
    move{i} = ScanFeature(move{i}, feature, tform);
end

initalGuess = [tform, cam];

Setup(1,metric, move, base, tform, cam, panFlag);

%% Evaluate metric and Optimize
Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();