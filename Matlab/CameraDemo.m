%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'GOM';

range = [0.5 0.5 0.5 10 10 10 40 0 0];
range(4:6) = pi*range(4:6)/180;
updatePeriod = 1;
dilate = 2;

%% Setup

%get scans and images
move = getPointClouds(1);
move{1} = move{1}(:,:);
base = getImagesC(1, false);

%get transform
tform = [-0.430764179648741,0.309541508254560,-0.0250793325294935,-1.55457292637707,-0.0415687455334417,3.09952627849462];

%get camera
cam = [765.669333685611, size(base{1}.v,2)/2,size(base{1}.v,1)/2];

initalGuess = [tform, cam];

Setup(metric, move, base, tform, cam, true);

%% Evaluate metric and Optimize
Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();