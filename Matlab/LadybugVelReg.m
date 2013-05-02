%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

global FIG
FIG.fig = figure;
FIG.count = 0;

%get ladybug parameters
ladybugParam = LadybugConfig;

%% input values
param = struct;

%options for swarm optimization
param.options = psooptimset('PopulationSize', 200,...
    'TolCon', 1e-1,...
    'StallGenLimit', 50,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf},...
    'SocialAttraction',1.25);

%how often to display an output frame
FIG.countMax = 500;


%range to search over (x, y ,z, rX, rY, rZ)
range = [0.3 0.3 0.5 5 5 5];
range(4:6) = pi*range(4:6)/180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'C:\Almond\';
%range of images to use
imRange = 1:100:3000;

%metric to use
metric = 'MI';

%number of times to run optimization
numTrials = 1;

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

[basePaths, movePaths, pairs] = MatchImageScan( path, imRange, true);

numBase = max(pairs(:,1));
numMove = max(pairs(:,2));

Initilize(numBase, numMove);

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
move = cell(numMove,1);
for i = 1:numMove
    move{i} = dlmread(movePaths{i},',');
    %move{i}(:,4) = sqrt(move{i}(:,1).^2 + move{i}(:,2).^2 + move{i}(:,3).^2);
    move{i} = getNorms(move{i},8);
    move{i}(:,4) = move{i}(:,4) - min(move{i}(:,4));
    move{i}(:,4) = move{i}(:,4) / max(move{i}(:,4));
	move{i}(:,4) = histeq(move{i}(:,4));
    
    m = filterScan(move{i}, metric, tform);
    LoadMoveScan(i-1,m,3);
    fprintf('loaded moving scan %i\n',i);
end

base = cell(numBase,1);
for i = 1:numBase
    idx2 = mod(i-1,5)+1;
    idx1 = (i - idx2)/5 + 1;
    baseIn = imread(basePaths{idx1,idx2});
    baseIn = imresize(baseIn,0.25);
    mask = imread([path 'Ladybug\masks\cam' int2str(idx2-1) '.png']);
    mask = imresize(mask,0.25);
    mask = mask(:,:,1);

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
    
    for q = 1:size(b,3)
        temp = b(:,:,q);
        temp(mask == 0) = 0;
        b(:,:,q) = temp;
    end
    LoadBaseImage(i-1,b);
    fprintf('loaded base image %i\n',i);
end

%% get image alignment
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    [tformOut, fOut]=pso(@(tform) alignLadyVel(base, move, pairs, tform, ladybugParam), 6,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,:) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal,1) / numTrials;
f = sum(fTotal) / numTrials;


fprintf('Final transform:\n     metric = %1.3f\n     rotation = [%2.2f, %2.2f, %2.2f]\n     translation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
        
%% cleanup
ClearLibrary;
rmPaths;