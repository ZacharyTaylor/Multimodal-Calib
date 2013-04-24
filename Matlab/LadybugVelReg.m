%% Setup
loadPaths;
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
tform = [-90 0 180 0 0 0];

%base path
path = 'C:\Data\Almond\';
%range of images to use
imRange = 27:28;

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

[basePaths, movePaths, pairs] = MatchImageScan( path, imRange );

numBase = max(pairs(:,1));
numMove = max(pairs(:,2));

Initilize(numBase, numMove);

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
move = cell(numMove,1);
for i = 1:numMove
    move{i} = dlmread(movePaths{i},',');
    m = filterScan(move{i}, metric);
    LoadMoveScan(i-1,m,3);
end

base = cell(numBase,1);
for i = 1:numBase
    idx2 = mod(i-1,5)+1;
    idx1 = (i - idx2)/5 + 1;
    baseIn = imread(basePaths{idx1,idx2});

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
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    [tformOut, fOut]=pso(@(tform) alignLadyVel(base, move, pairs, tform), 6,[],[],[],[],param.lower,param.upper,[],param.options);

    tformTotal(i,:) = tformOut;
    fTotal(i) = fOut;
end

tform = sum(tformTotal,1) / numTrials;
f = sum(fTotal) / numTrials;


fprintf('Final transform:\n     metric = %1.3f\n     translation = [%2.2f, %2.2f, %2.2f]\n     rotation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
        
%% cleanup
ClearLibrary;
rmPaths;