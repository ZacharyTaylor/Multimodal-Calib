% Demo of program aligning a series of velodyne scans with a series of
% images

% This demo requires drive 35 of the KITTI dataset it can be found at
% www.mrt.kit.edu/geigerweb/cvlibs.net/kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip

%% User set parameters

%inital guess as to the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rx, ry, rz)
tform = [0.0286078768104492,-0.0859292877386722,-0.218274737581129,-1.56005188433452,0.0109412403037259,-1.56849213340563];

%camera intrinsic parameters (taken from calibration of camera 0 given on 
%the kitti site)
cam = [7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00;...
    0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00;...
    0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00];

%Number of scans to use in calibration (130 scans in drive 35 set, must 
%fit in gpu ram. For kitti data need about 10 mb per scan-image
%pair. I usually find 20 is enough for a good result)
scanIdx = 3:120;

%True for panoramic camera, false otherwise
panFlag = false;

%Path to Kitti dataset
kittiPath = 'C:\DataSets\Mobile Sensor Plaforms\KITTI\raw data\drive 35';

%% Setup

metric = 'None';

%number of moving scans
numMove = length(scanIdx);
%number of base images
numBase = length(scanIdx);

%get scans and images
folder = [kittiPath '\image_02\data\'];
files = dir(folder);
fileIndex = find(~[files.isdir]);

base = cell(length(scanIdx),1);
for i = 1:length(scanIdx)
    fileName = files(fileIndex(scanIdx(i))).name;
    base{i} = imread([folder fileName]);
    base{i}(:,:,1) = medfilt2(base{i}(:,:,1),[5 5]);
    base{i}(:,:,2) = medfilt2(base{i}(:,:,2),[5 5]);
    base{i}(:,:,3) = medfilt2(base{i}(:,:,3),[5 5]);
end
move = ReadKittiVelData( kittiPath, scanIdx);

for i = 1:length(move)
    move{i} = move{i}(sqrt(sum(move{i}(:,1:3).^2,2)) < 40,:);
end

Setup(1,metric, move, base, tform, cam, panFlag);

%% Colour and align

out = cell(size(move));
for i =1:length(scanIdx);
    i
    out{i} = ColourScan(i-1);
end

%merge
lidarT = icp( out, -1.4, scanIdx );
lidarP = lidarT;
for i =1:length(scanIdx);
    if(i ~= 1)
        lidarP{i} = lidarP{i-1}/lidarT{i};
    end
    temp = out{i}(:,1:4);
    temp(:,4) = 1;
    temp = (lidarP{i}*(temp'))';
    out{i}(:,1:3) = temp(:,1:3);
end

res = cell2mat(out);

%% Clean up
ClearEverything();