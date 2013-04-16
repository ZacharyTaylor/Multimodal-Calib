%% Setup
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

Initilize(1,1);

SetupMIMetric();
SetupAffineTform();

global FIG
FIG.fig = figure;
FIG.count = 0;

%% input values

param = struct;
%options for swarm optimization
param.options = psooptimset('PopulationSize', 80,...
    'TolCon', 1e-1,...
    'StallGenLimit', 30,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf},...
    'SocialAttraction',1.25);

FIG.countMax = 100;

range = [30 30 1 0.1 0.1 0.01 0.01];

initalGuess = [0 0 0 1 1 0 0];

param.lower = initalGuess - range;
param.upper = initalGuess + range;

%% get Data
[base, move] = getImages();

%normalize
base = single(base)/255;
move = single(move)/255;

%hestogram equalize
base = histeq(base);
move = histeq(move);

%% get image alignment

LoadMoveImage(0,move);
LoadBaseImage(0,base);

[tform, ~]=pso(@(tform) alignImages(base, move, tform), 7,[],[],[],[],param.lower,param.upper,[],param.options);

        

%% display result


%% clean up
clearTexture(moveTexPtr);
clearImage(baseD);
unloadlibrary('cudaImage');