%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_LEVEL
DEBUG_LEVEL = 1;

if(~exist('FIG','var'))
    global FIG
    FIG.fig = figure;
    FIG.count = 0;
end

%% input values

%how often to display an output frame
FIG.countMax = 0;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = [-3.3562 -0.86656 0.48898 -1.4888 -0.043498 2.3323 762.04];
%tform(4:6) = pi.*tform(4:6)./180;

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move scan]
pairs = [1 1];

%if camera panoramic
panoramic = 1;

%% setup transforms and images
SetupCamera(panoramic);

SetupCameraTform();

Initilize(numMove,numBase);

%% get Data

%if(~exist('move','var'))
    %move = getPointClouds(numMove);
    
%end
%base = getImagesC(numBase, false);

for i = 1:numMove
    m = single(move{i});
    LoadMoveScan(i-1,m,3);
end

for i = 1:numBase
    b = single(base{i}.c)/255;
    LoadBaseImage(i-1,b);
end

%% get image alignment
cameraMat = cam2Pro(tform(7),tform(7),size(base{1}.v,2)/2,size(base{1}.v,1)/2);
SetCameraMatrix(cameraMat);

%get transformation matirx       
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

SetTformMatrix(tformMat);
Transform(pairs(i,2)-1);
InterpolateBaseValues(pairs(i,2)-1);

out = GetGen();
out = [m(:,1:3), out(:,4:end)];

%remove black points
%out = out(any(out(:,4),2),:);

%set range
%out(:,4:end) = out(:,4:end) - repmat(min(out(:,4:end)),size(out,1),1);
%out(:,4:end) = out(:,4:end) ./ repmat(max(out(:,4:end)),size(out,1),1);

clear output;
output.vertex.x = out(:,1);
output.vertex.y = out(:,2);
output.vertex.z = out(:,3);
output.vertex.red = out(:,4);
output.vertex.green = out(:,5);
output.vertex.blue = out(:,6);
ply_write(output,'out.ply','binary_big_endian');

%% cleanup
ClearLibrary;
rmPaths;