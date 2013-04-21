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

FIG.countMax = 500;

range = [30 30 10 0.1 0.1 0.01 0.01];

tform = [0 0 0 1 1 0 0];

numMove = 1;
numBase = 1;
pairs = [1 1];

sigma = 0;

numTrials = 10;

metric = 'MI';

SetupAffineTform();

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

%G = fspecial('gaussian',[50 50],sigma);

%% get Data
move = getImagesStruct(numMove);

for i = 1:numMove
    if(strcmp(metric,'MI'))
        m = single((move{i}.v))/255;
    elseif(strcmp(metric,'GOM'))   
        m = single((move{i}.v))/255;
        [mag,phase] = imgradient(m);
        
%         mag = imfilter(mag,G,'same');
%         mag = mag/max(mag(:));
%         phase = imfilter(phase,G,'same');

        m = zeros([size(mag) 2]);
        m(:,:,1) = mag;
        m(:,:,2) = phase;
    else
        error('Invalid metric type');
    end
    
    LoadMoveImage(i-1,m);
end

base = getImagesStruct(numBase);

for i = 1:numBase
    if(strcmp(metric,'MI'))
        b = single((base{i}.v))/255;
    elseif(strcmp(metric,'GOM'))
        b = single((base{i}.v))/255;
        [mag,phase] = imgradient(b);
        
%         mag = imfilter(mag,G,'same');
%         mag = mag/max(mag(:));
%         phase = imfilter(phase,G,'same');
        
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
    [tformOut, fOut]=pso(@(tform) alignImages(base, move, pairs, tform), 7,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,:) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal,1) / numTrials;
f = sum(fTotal) / numTrials;

 fprintf('Final transform:\n     metric = %1.3f\n     translation = [%3.0f, %3.0f]\n     rotation = %1.2f\n     scale = [%1.2f,%1.2f]\n     shear = [%0.3f, %0.3f]\n\n',...
            f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6),tform(7));
        