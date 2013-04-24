%% Setup
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
param.options = psooptimset('PopulationSize', 200,...
    'TolCon', 1e-1,...
    'StallGenLimit', 30,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf},...
    'SocialAttraction',1.25);

%how often to display an output frame
FIG.countMax = 0;

%range to search over (rX, rY, rZ, x, y ,z)
range = [30 30 30 0.1 0.1 0.01];

%inital guess of parameters (rX, rY, rZ, x, y ,z) (rotate then translate,
%rotation order ZYX)
initalGuess = [-90 0 180 0 0 0];

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move scan]
pairs = [1 1];

%metric to use
metric = 'MI';

%if camera panoramic
panoramic = 1;

%camera parameters (focal length, fX, fY)
camera = [1000 500 500];

%number of times to run optimization
numTrials = 2;


%% setup transforms and images
SetupCamera(panoramic);

cameraMat = cam2Pro(camera(1),camera(1),camera(2),camera(3));
SetCameraMatrix(cameraMat);
SetupCameraTform();

Initilize(numMove,numBase);

param.lower = initalGuess - range;
param.upper = initalGuess + range;

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
    if(strcmp(metric,'MI'))
        m = single(move{i});
    elseif(strcmp(metric,'GOM'))   
        m = single(move{i});
        [mag,phase] = getGradient(m);
        m = [m(:,1:3),mag,phase];
    else
        error('Invalid metric type');
    end
    
    LoadMoveScan(i-1,m,3);
end

base = getImagesStruct(numBase);

for i = 1:numBase
    if(strcmp(metric,'MI'))
        b = single(base{i}.v)/255;
    elseif(strcmp(metric,'GOM'))
        b = single(base{i}.v)/255;
        [mag,phase] = imgradient(b);
        b = zeros([size(mag) 2]);
        b(:,:,1) = mag;
        b(:,:,2) = phase;
    else
        error('Invalid metric type');
    end
    
    LoadBaseImage(i-1,b);
end



%% get image alignment
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    [tformOut, fOut]=pso(@(tform) alignPoints(base, move, pairs, tform), 6,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,:) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal,1) / numTrials;
f = sum(fTotal) / numTrials;


fprintf('Final transform:\n     metric = %1.3f\n     translation = [%2.2f, %2.2f, %2.2f]\n     rotation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
        