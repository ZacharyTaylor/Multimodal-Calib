%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(1,1);

SetupAffineTform();

global FIG
FIG.fig = figure;
FIG.count = 0;
FIG.countMax = 0;

%% input values

%transform to apply (x, y, r, sx, sy, kx, ky)
tform = [0 0 0 1 1 0 0];

%metric to use
metric = 'MI';

%% setup Metric
if(strcmp(metric,'MI'))
    SetupMIMetric();
elseif(strcmp(metric,'GOM'))   
    SetupGOMMetric();
else
    error('Invalid metric type');
end

%% get Data
move = getImagesC(1);
base = getImagesC(1);

%% setup Metric
m = filterImage(move{1}, metric);
b = filterImage(base{1}, metric);
    
%% get image alignment
LoadMoveImage(0,m);
LoadBaseImage(0,b);

alignImages(base, move, [1,1], tform);

%% cleanup
ClearLibrary;
rmPaths;