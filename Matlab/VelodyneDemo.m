% Demo of program aligning a series of velodyne scans with a series of
% images

% This demo requires drive 35 of the KITTI dataset it can be found at
% www.mrt.kit.edu/geigerweb/cvlibs.net/kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip

%% User set parameters

%metric to use, can be one of the following
% SSD - normalized SSD metric
% MI - mutual information (uses a histogram method with 50 bins)
% NMI - normalized mutual information (uses same method as above)
% GOM - gradient orientation measure
% GOMS - modified GOM method that evalutes statistical significants of
% results. Usually outperforms GOM
% LEV - levinson's method (Buggy, still working on it)
% None - no metric assigned used in generating images and coloured scans
% Note (NMI, MI, LEV and GOM) multiplied by -1 to give minimums in
% optimization
metric = 'GOM';

%inital guess as to the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rx, ry, rz)
tform = [0,0,0,-pi/2,0,-pi/2];

%camera intrinsic parameters (taken from calibration of camera 0 given on 
%the kitti site)
cam = [7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00;...
    0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00;...
    0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00];

%range around transform where the true solution lies 
%can be the following forms
%[x,y,z]
%[x,y,z,rx,ry,rz]
%[x,y,z,rx,ry,rz,f,cx,cy] if camera of form [f,cx,cy]
%[x,y,z,rx,ry,rz,fx,fy,cx,cy] if camera of form [fx,fy,cx,cy]
%rotations in radians, rotation order rx, ry, rz)
range = [0.5 0.5 0.5 0.1 0.1 0.1];

%Sets the update rate in seconds of the output that can be used to evaluate
%the metrics progress. Updating involves transfering the whole image off
%the gpu and so for large scans causes a significant slow down (increase
%value to reduce this issue)
updatePeriod = 30;

%How much to dialate each point by when generating an image from it (only
%effects view generated in updates, does not change metric values)
dilate = 3;

%Number of scans to use in calibration (130 scans in drive 35 set, must 
%fit in gpu ram. For kitti data need about 10 mb per scan-image
%pair. I usually find 20 is enough for a good result)
numScans = 10;

%feature to use as intensity information of lidar scans. Options are 
%intensity - basic lidar intensity
%range - distance of points from the lidar
%normals - angle between line from lidar to point and a horizontal plane
feature = 'intensity';

%True for panoramic camera, false otherwise
panFlag = false;

%Path to Kitti dataset
kittiPath = 'C:\Users\Zachary\Documents\Datasets\Kitti\2011_09_26_drive_0035_sync';

%% Setup

%number of moving scans
numMove = numScans;
%number of base images
numBase = numScans;

%get scans and images
folder = [kittiPath '\image_00\data\'];
files = dir(folder);
fileIndex = find(~[files.isdir]);

scanIdx = randperm(length(fileIndex));
scanIdx = sort(scanIdx(1:numScans));

base = cell(length(scanIdx),1);
for i = 1:length(scanIdx)
    fileName = files(fileIndex(scanIdx(i))).name;
    base{i} = imread([folder fileName]);
end
move = ReadKittiVelData( kittiPath, scanIdx);

%get features for scans
for i = 1:size(move,1)
    move{i} = ScanFeature(move{i}, feature, tform);
end

initalGuess = tform;

Setup(10,metric, move, base, tform, cam, panFlag);

%% Evaluate metric and Optimize
%Align( initalGuess, 0, dilate );
Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();