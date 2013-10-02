SetupLib();

%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;

%% Setup
Initilize();

%get scans and images
move = getPointClouds(1);
base = getImagesC(1, false);

%filter scans and images


%get transform
tform = cell(1,1);
tform{1} = [0.092061 0.15907 -0.3949 -1.549 -0.036013 3.0793];

%get camera
cam = cell(1,1);
cam{1} = [750, size(base{1}.v,2),size(base{1}.v,1)];

%% Transfer to GPU
%load scans
for i = 1:size(move,1)
    AddMovingScan(move{i}(:,1:3), move{i}(:,end));
end
%load images
for i = 1:size(base,1)
    AddBaseImage(base{i}.v);
end
%load tforms
for i = 1:size(tform,1)
    AddTform(tform{i});
end
%load cameras
for i = 1:size(cam,1)
    AddCamera(cam{i}, false);
end

%set indecies
SetSingleScanIndex();

EvalMetric();

%% Clean up
ClearEverything();
unloadlibrary('LibCal');