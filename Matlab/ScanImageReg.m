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
param.options = psooptimset('PopulationSize', 500,...
    'TolCon', 1e-10,...
    'StallGenLimit', 100,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf});

%how often to display an output frame
FIG.countMax = 5000000000;

%range to search over (x, y ,z, rX, rY, rZ)
range = [1 1 1 5 5 5 40];
range(4:6) = pi.*range(4:6)./180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
%tform = [0 0 0 -90 0 68 780];
%tform = [0 0 0 -90 -2 177 780];
%tform = [0 0 0 -91 1 295 780];
tform = [0 0 0 -90 1 317 780];
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
numTrials = 10;


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
    %base{i}.v = uint8(255*b(:,:,2)/90);
end



%% get image alignment
tformTotal = zeros(numTrials,2*size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    tformIn = tform + (rand(1,7)-0.5).*range;
    
    param.lower = tformIn - range;
    param.upper = tformIn + range;

    [tformOut, fOut]=pso(@(tformIn) alignPoints(base, move, pairs, tformIn), 7,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,1:7) = tformIn;
    tformTotal(i,8:14) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal(:,8:14),1) / numTrials;
f = sum(fTotal) / numTrials;


fprintf('Final transform:\n     metric = %1.3f\n     translation = [%2.2f, %2.2f, %2.2f]\n     rotation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
 
%% cleanup
ClearLibrary;
rmPaths;