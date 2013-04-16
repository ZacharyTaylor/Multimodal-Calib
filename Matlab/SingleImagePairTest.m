%% Setup
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

Initilize(1,1);

SetupMIMetric();
SetupAffineTform();

%% input values

tform = [0 0 0 1 1 0 0];

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

alignImages(base, move, tform);

        

%% display result


%% clean up
clearTexture(moveTexPtr);
clearImage(baseD);
unloadlibrary('cudaImage');