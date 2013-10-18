%% User set parameters

%number of moving scans
numMove = 1;
%number of base images
numBase = 1;
%metric to use
metric = 'NMI';

range = [2 2 2 3 3 3 300 0 0];
range(4:6) = pi*range(4:6)/180;
updatePeriod = 5;
dilate = 4;

%% Setup

%get scans and images
move = getPointClouds(1);
move{1} = move{1} - repmat([10455.289,7824.418,681.748 0],size(move{1},1),1);
%move{1} = move{1}(randi(size(move{1},1),500000,1),:);
move{1} = move{1}(:,:);
base = getImagesC(1, false);
%base{1}.v = imresize(base{1}.v,0.25);

%get transform
tform = [0 0 0 -83.8 1.8 48];
tform(4:6) = pi*tform(4:6)/180;

%move{1} = GetNorms(move{1},tform);

%get camera
cam = [5340, size(base{1}.v,2)/2,size(base{1}.v,1)/2];

initalGuess = [tform, cam];

Setup(metric, move, base, tform, cam, true);

%% Evaluate metric and Optimize
%Align(tform, updatePeriod, dilate);
Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();