%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'GOM';

range = [10 10 5 0.1 0.1 0.01 0.01];
updatePeriod = 1;
dilate = 2;

%% Setup

%get scans and images
move = getImagesC(1, false);
base = getImagesC(1, false);

%get transform
tform = [0 0 0 1 1 0 0];
initalGuess = tform;

Setup(metric, move, base, tform);

%% Evaluate metric and Optimize
Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();