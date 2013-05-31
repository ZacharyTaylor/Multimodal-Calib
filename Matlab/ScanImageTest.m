%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

global FIG
FIG.fig = figure;
FIG.count = 0;

%% input values
param = struct;

%options for swarm optimization
param.options = psooptimset('PopulationSize', 300,...
    'TolCon', 1e-1,...
    'StallGenLimit', 30,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf});

%how often to display an output frame
FIG.countMax = 0;

%range to search over (x, y ,z, rX, rY, rZ)
range = [1 1 1 10 10 10 50];
range(4:6) = pi.*range(4:6)./180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = [0 0 0 0 0 0 1000];

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move scan]
pairs = [1 1];

%metric to use
metric = 'MI';

%if camera panoramic
panoramic = 1;

%% setup transforms and images
SetupCamera(panoramic);

SetupCameraTform();

Initilize(numMove,numBase);

%% setup Metric
if(strcmp(metric,'MI'))
    SetupMIMetric();
elseif(strcmp(metric,'GOM'))   
    SetupGOMMetric();
else
    error('Invalid metric type');
end

%% get Data

move = getPointClouds(numMove);

for i = 1:numMove
    m = filterScan(move{i}, metric, tform);
    LoadMoveScan(i-1,m,3);
end

base = getImagesC(numBase, true);

for i = 1:numBase
    b = filterImage(base{i}, metric);
    LoadBaseImage(i-1,b);
end

%% get image alignment
f = alignPoints(base, move, pairs, tform);    

%% cleanup
ClearLibrary;
rmPaths;