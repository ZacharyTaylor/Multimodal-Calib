%% Setup
loadPaths;
%set(0,'DefaultFigureWindowStyle','docked');
clc;
close all;

global DEBUG_LEVEL
DEBUG_LEVEL = 1;

global FIG
FIG.fig = figure;
FIG.count = 0;

%get ladybug parameters
ladybugParam = ShrimpConfig;
ladybugParam = ladybugParam.Ladybug;

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
FIG.countMax = 4;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'C:\DataSets\Mobile Sensor Plaforms\Shrimp\Apples\';
%range of images to use
imRange = 1500;


%metric to use
metric = 'MI';

%feature to use (return, distance, normals)
feature = 'normals';

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

[basePaths, movePaths, pairs] = MatchImageScan( path, imRange, true );

numBase = max(pairs(:,1));
numMove = max(pairs(:,2));

Initilize(numBase, numMove);

%% get Data
move = cell(numMove,1);
for i = 1:numMove
    move{i} = ReadVelData(movePaths{i});
end

base = cell(numBase,1);
for i = 1:numBase
    idx2 = mod(i-1,5)+1;
    idx1 = (i - idx2)/5 + 1;
    baseIn = imread(basePaths{idx1,idx2});

    for q = 1:size(baseIn,3)
        temp = baseIn(:,:,q);
        temp(temp ~= 0) = histeq(temp(temp ~= 0));
        baseIn(:,:,q) = temp;
    end
    
    if(size(baseIn,3)==3)
        base{i}.c = baseIn;
        base{i}.v = rgb2gray(baseIn);
    else
        base{i}.c = baseIn(:,:,1);
        base{i}.v = baseIn(:,:,1);
    end
end

%% setup Metric
if(strcmp(metric,'MI'))
    SetupMIMetric();
elseif(strcmp(metric,'GOM'))   
    SetupGOMMetric();
else
    error('Invalid metric type');
end

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

%% filter data
for i = 1:numMove
    m = filterScan(move{i}, metric, tform);
    LoadMoveScan(i-1,m,3);
end

for i = 1:numBase
    b = filterImage(base{i}, metric);
    LoadBaseImage(i-1,b);
end

%% get image alignment
alignLadyVel(base, move, pairs, tform, ladybugParam);
        
%% cleanup
ClearLibrary;
rmPaths;