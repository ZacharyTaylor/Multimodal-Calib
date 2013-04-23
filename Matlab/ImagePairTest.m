%% Setup
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

tform = [0 0 0 1 1 0 0];
metric = 'GOM';



%% get Data
move = getImagesStruct(1);
base = getImagesStruct(1);

m = single(move{1}.v)/255;
b = single(base{1}.v)/255;

%% setup Metric
if(strcmp(metric,'MI'))
    
    SetupMIMetric();
    
elseif(strcmp(metric,'GOM'))
    
    [mag,phase] = imgradient(m);
    m(end,end,2) = 0;
    m(:,:,1) = mag;
    m(:,:,2) = phase;
    
    [mag,phase] = imgradient(b);
    b(end,end,2) = 0;
    b(:,:,1) = mag;
    b(:,:,2) = phase;
    
    SetupGOMMetric();
    
else
    
    error('Invalid metric type');
    
end
    
%% get image alignment


LoadMoveImage(0,m);
LoadBaseImage(0,b);

alignImages(base, move, [1,1], tform);

%% clean up

ClearLibrary();