%% Setup
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

Initilize(1,1);

SetupMIMetric();
SetupAffineTform();

%% input values

tform = [0 0 0 3 2 0 0];

%% get Data
[base, move] = getImages();

base = single(base);
move = single(move);

%% get image alignment

LoadMoveImage(0,move);
LoadBaseImage(0,base);

alignImages(base, move, tform);