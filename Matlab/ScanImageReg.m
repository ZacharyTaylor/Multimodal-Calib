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
FIG.countMax = 50;

%range to search over (x, y ,z, rX, rY, rZ)
range = [1 1 1 10 10 10 50];
range(4:6) = pi.*range(4:6)./180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = [0 0 0 -90 0 68 800];
tform(4:6) = pi.*tform(4:6)./180;

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move scan]
pairs = [1 1];

%metric to use
metric = 'MI';

%if camera panoramic
panoramic = 1;

%number of times to run optimization
numTrials = 1;


%% setup transforms and images
SetupCamera(panoramic);

SetupCameraTform();

Initilize(numMove,numBase);

param.lower = tform - range;
param.upper = tform + range;

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
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    [tformOut, fOut]=pso(@(tform) alignPoints(base, move, pairs, tform), 7,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,:) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal,1) / numTrials;
f = sum(fTotal) / numTrials;


fprintf('Final transform:\n     metric = %1.3f\n     translation = [%2.2f, %2.2f, %2.2f]\n     rotation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
 
%% cleanup
ClearLibrary;
rmPaths;