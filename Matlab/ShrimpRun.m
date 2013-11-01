% Aligns images and scans produced by the ACFR's shrimp 
% This file should not be in the public repo, however if I have forgotten
% to exlcude it just ignore it as it uses functions and datasets not 
% avaliable to the public

%% User set parameters

%metric to use, can be one of the following
% SSD - normalized SSD metric
% MI - mutual information (uses a histogram method with 50 bins)
% NMI - normalized mutual information (uses same method as above)
% GOM - gradient orientation measure
% GOMS - modified GOM method that evalutes statistical significants of
% results. Usually outperforms GOM
% LEV - levinson's method (not yet implemented)
% None - no metric assigned used in generating images and coloured scans
% Note (NMI, MI, LEV and GOM) multiplied by -1 to give minimums in
% optimization
metric = 'GOMS';

%range around transform where the true solution lies 
%can be the following forms
%[x,y,z]
%[x,y,z,rx,ry,rz]
%[x,y,z,rx,ry,rz,f,cx,cy] if camera of form [f,cx,cy]
%[x,y,z,rx,ry,rz,fx,fy,cx,cy] if camera of form [fx,fy,cx,cy]
%rotations in radians, rotation order rx, ry, rz)
range = [0.5 0.5 0.5 3 3 5];
range(4:6) = (pi/180)*range(4:6);

%Sets the update rate in seconds of the output that can be used to evaluate
%the metrics progress. Updating involves transfering the whole image off
%the gpu and so for large scans causes a significant slow down (increase
%value to reduce this issue)
updatePeriod = 5;

%How much to dialate each point by when generating an image from it (only
%effects view generated in updates, does not change metric values)
dilate = 4;

%Number of scans to use in calibration
scanIdx = 50;%500:50:900;

%True for panoramic camera, false otherwise
panFlag = false;

%Path to Kitti dataset
shrimpPath = 'C:\DataSets\Mobile Sensor Plaforms\Shrimp\Almond';

%% Setup
Shrimp = ShrimpConfig('C:\Sensors\shrimpCalib.config');

velTform = Shrimp.velodyne.offset;
velTform = CreateTformMat( velTform );

cam = cell(1,5);
for i = 1:5
    cam{1,i} = [Shrimp.ladybug.(['camera_' num2str(i)]).focal_length, Shrimp.ladybug.(['camera_' num2str(i)]).centre'];
end

%initalGuess = inv(CreateTformMat(Shrimp.ladybug.offset))*velTform;
initalGuess = [0.0743762712305078,-0.0558455905701451,0.216919292307192,-0.0331852358287054,3.11052187589235,5.08660829095249];
%initalGuess(4:6) = (pi/180)*initalGuess(4:6);

tform = cell(1,5);
multiCamTform = cell(1,5);
for i = 1:5
    multiCamTform{1,i} = Shrimp.ladybug.(['camera_' num2str(i)]).offset';
    tform{1,i} = CreateTformMat(multiCamTform{1,i})\CreateTformMat(initalGuess);
end

%get scans and images
% folder = [shrimpPath '\LadybugColourVideo\cam0\'];
% files = dir(folder);
% fileIndex = find(~[files.isdir]);
% scanIdx = randperm(length(fileIndex));
% scanIdx = sort(scanIdx(1:numScans));

[ base, move, ~ ] = MatchImageScan(shrimpPath, scanIdx, false);

for i = 1:length(base(:))
    base{i} = imread(base{i});
end
for i = 1:length(move(:))
    [move{i},~,~] = ReadVelData(move{i});
end

Setup(10,metric, move, base, tform, cam, panFlag);

%% Evaluate metric and Optimize
%Optimize( initalGuess, range, updatePeriod, dilate, multiCamTform )
Align(initalGuess, 0, dilate, multiCamTform);
%RES = ColourScan(0);

%% Clean up
ClearEverything();