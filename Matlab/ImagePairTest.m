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

tform = [0 0 0 1 1 0 0];

%% get Data
[base, move] = getImages();

%% get image alignment
move = single(move)/255;
base = single(base)/255;

LoadMoveImage(0,move);
LoadBaseImage(0,base);

alignImages(base, move, [1,1], tform);