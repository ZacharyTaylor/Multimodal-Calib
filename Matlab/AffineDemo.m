% Demo of aligning two images

%% User set parameters

%metric to use, can be one of the following
% SSD - normalized SSD metric
% MI - mutual information (uses a histogram method with 50 bins)
% NMI - normalized mutual information (uses same method as above)
% GOM - gradient orientation measure
% GOMS - modified GOM method that evalutes statistical significants of
% results. Usually outperforms GOM
% LEV - levinson's method (not yet implemented)
% None - no metric assigned used in generating images and coloured scans
% Note (NMI, MI, LEV and GOM) multiplied by -1 to give minimums in
% optimization
metric = 'GOM';

%inital guess as to the transform between the camera and the lidar
%can be either a 3x3 transform matrix or 
%[trans in x, trans in y, rotation, scale x, scale y, shear x, shear y]
tform = [0 0 0 1 1 0 0];

%range around transform where the true solution lies 
%range = [50 50 0.2 0.1 0.1 0.01 0.01];

%Sets the update rate in seconds of the output that can be used to evaluate
%the metrics progress. Updating involves transfering the whole image off
%the gpu and so for large scans causes a significant slow down (increase
%value to reduce this issue)
updatePeriod = 1;

%How much to dialate each point by when generating an image from it (only
%effects view generated in updates, does not change metric values)
dilate = 2;

%% Setup

%get images
in = GetImagesC(2, false);
base = in(2);
move = in(1);

initalGuess = tform;

Setup(1,metric, move, base, tform);

%% Evaluate metric and Optimize
Align( initalGuess, updatePeriod, dilate );
%Optimize( initalGuess, range, updatePeriod, dilate )

%% Clean up
ClearEverything();