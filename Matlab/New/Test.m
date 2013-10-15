%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'GOM';

range = [0.5 0.5 0.5 0.3 0.3 0.3];
updatePeriod = 1;
dilate = 2;

%% Setup

%get scans and images
move = getPointClouds(1);
move{1} = move{1}(:,:);
base = getImagesC(1, false);

%get transform
tform = [0.092061 0.15907 -0.3949 -1.549 -0.036013 3.0793];
baseTform = tform;
initalGuess = tform;

%get camera
cam = [750, size(base{1}.v,2)/2,size(base{1}.v,1)/2];

Setup(metric, move, base, tform, cam, false, baseTform);

%% Evaluate metric and Optimize
Optimize( initalGuess, range, updatePeriod, dilate )

imshow(image);
drawnow;

%% Clean up
ClearEverything();