%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'None';

%% Setup

%get scans and images
move = getPointClouds(1);
move{1} = move{1}(:,:);
move{1} = move{1} - repmat([10455.289,7824.418,681.748 0],size(move{1},1),1);
base = getImagesC(1, false);

%get transform
tform = [0 0 0 -83.8 1.8 48];
tform(4:6) = pi*tform(4:6)/180;

%get camera
cam = [5340, size(base{1}.v,2)/2,size(base{1}.v,1)/2];

Setup(metric, move, base, tform, cam, true);

%% Evaluate metric and Optimize
scan = ColourScan(0);

dlmwrite('ScanOut.csv',scan,'precision',12 );

%% Clean up
ClearEverything();