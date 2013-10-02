%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_LEVEL
DEBUG_LEVEL = 1;

global FIG
FIG.fig = figure;
FIG.count = 0;

%% input values
param = struct;

%options for swarm optimization
param.options = psooptimset('PopulationSize', 2000,...
    'TolCon', 1e-1,...
    'StallGenLimit', 30,...
    'Generations', 300,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf});

%how often to display an output frame
FIG.countMax = 100000;

%range to search over (x, y ,z, rX, rY, rZ)
range = [4 4 0.5 5 5 10 80];
range(4:6) = pi.*range(4:6)./180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = [0.092061 0.15907 -0.3949 -1.549 -0.036013 3.0793 749.94];%[0 0 5 -90 -2.3 0 770];
%tform(4:6) = pi.*tform(4:6)./180;

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move scan]
pairs = [1 1];

%metric to use
metric = 'GOM';

%feature to use (return, distance, normals)
feature = 'return';

%if camera panoramic
panoramic = 1;

%number of times to run optimization
numTrials = 1;


%% setup transforms and images
SetupCamera(panoramic);

SetupCameraTform();

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

%% get Data
move = getPointClouds(numMove);
base = getImagesC(numBase, true);

%% setup feature
if(strcmp(feature,'return'))
    %already setup
elseif(strcmp(feature,'distance'))   
    for i = 1:numMove
        move{i}(:,4) = sqrt(sum(move{i}(:,1:3).^2,2));
        move{i}(:,4) = move{i}(:,4) - min(move{i}(:,4));
        move{i}(:,4) = move{i}(:,4) / max(move{i}(:,4));
        move{i}(:,4) = 1-histeq(move{i}(:,4));
    end
elseif(strcmp(feature,'normals'))
    for i = 1:numMove
        move{i} = getNorms(move{i}, tform);
    end
else
    error('Invalid feature type');
end

for i = 1:numMove
    m = filterScan(move{i}, metric, tform);
    LoadMoveScan(i-1,m,3);
end

for i = 1:numBase
    b = filterImage(base{i}, metric);
    LoadBaseImage(i-1,b);
end

%% get image alignment
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

j = 0;
val = zeros(101,101);
for y = -0.5:0.01:0.5
    j = j+1
    i = 0;
    for x = -0.5:0.01:0.5
        i = i+1;
            val(i,j) = -alignPoints(base, move, pairs, [0.092061 0.15907 -0.3949 -1.549 -0.036013+pi*y/2 3.0793+pi*x/2 749.94]);
    end
end
surf(val);
% for i = 1:numTrials
%     [tformOut, fOut]=pso(@(tform) alignPoints(base, move, pairs, tform), 7,[],[],[],[],param.lower,param.upper,[],param.options);
% 
%     tformTotal(i,:) = tformOut;
%     fTotal(i) = fOut;
% end
% 
% tform = sum(tformTotal,1) / numTrials;
% f = sum(fTotal) / numTrials;
% 
% tform(4:6) = pi.*tform(4:6)./180;
% 
% fprintf('Final transform:\n     metric = %1.3f\n     translation = [%2.2f, %2.2f, %2.2f]\n     rotation = [%2.2f, %2.2f, %2.2f]\n\n',...
%             f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6));
 
%% cleanup
ClearLibrary;
rmPaths;