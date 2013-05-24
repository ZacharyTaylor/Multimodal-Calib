%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','docked');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

global FIG
%FIG.fig = figure;
FIG.count = 0;

%get ladybug parameters
ladybugParam = FordConfig;

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

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'C:\Data\Ford\mi-extrinsic-calib-data\';
%range of images to use
imRange = [13];


%metric to use
metric = 'MI';

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

%[basePaths, movePaths, pairs] = MatchImageScan( path, imRange, true );
[basePaths, movePaths, pairs] = MatchFord( path, imRange, true );

numBase = max(pairs(:,1));
numMove = max(pairs(:,2));

Initilize(numBase, numMove);

%% setup Metric
if(strcmp(metric,'MI'))
    SetupMIMetric();
elseif(strcmp(metric,'GOM'))   
    SetupGOMMetric();
else
    error('Invalid metric type');
end

%% get Data
move = cell(numMove,1);
for i = 1:numMove
    move{i} = dlmread(movePaths{i},' ',1,0);

    %move{i}(:,4) = sqrt(move{i}(:,1).^2 + move{i}(:,2).^2 + move{i}(:,3).^2);
    %move{i} = getNorms(move{i},8);
    %move{i}(:,4) = 1-histeq(move{i}(:,4));

    m = filterScan(move{i}, metric);
    
    m(:,4) = m(:,4) - min(m(:,4));
    m(:,4) = m(:,4) / max(m(:,4));
    
    LoadMoveScan(i-1,m,3);
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
            
    b = filterImage(base{i}, metric);
    LoadBaseImage(i-1,b);
end

%% get image alignment
alignLadyVel(base, move, pairs, tform, ladybugParam);


        
%% cleanup
ClearLibrary;
rmPaths;