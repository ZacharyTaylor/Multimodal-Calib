%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

global FIG
FIG.fig = figure;
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure. 
FIG.count = 0;

%get ladybug parameters
ladybugParam = LadybugConfig;

%% input values
param = struct;

%options for swarm optimization
param.options = psooptimset('PopulationSize', 200,...
    'TolCon', 1e-10,...
    'StallGenLimit', 50,...
    'Generations', 200,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf});

%how often to display an output frame
FIG.countMax = 5000;


%range to search over (x, y ,z, rX, rY, rZ)
range = [0.5 0.5 0.5 2 2 2];
range(4:6) = pi*range(4:6)/180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'G:\DataSets\Mobile Sensor Plaforms\Shrimp\Almond\';
%range of images to use
imRange = sort(1+ round(253*rand(20,1)))'
%metric to use
metric = 'GOM';

scale = 0.5;

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
elseif(strcmp(metric,'LIV'))
    SetupLIVMetric();
else
    error('Invalid metric type');
end

%% get move{i}
move = cell(numMove,1);
for i = 1:numMove
    %move{i} = dlmread(movePaths{i},' ');
    move{i} = ReadVelData(movePaths{i});

%     
% 
 %move{i}(:,4) = sqrt(move{i}(:,1).^2 + move{i}(:,2).^2 + move{i}(:,3).^2);
 move{i} = getNorms(move{i},tform);
% %     
   
% sphere = zeros(size(move{i},1),2);
% sphere(:,1) = atan2(move{i}(:,1), move{i}(:,2));
% sphere(:,2) = atan(move{i}(:,3)./ sqrt(move{i}(:,1).^2 + move{i}(:,2).^2));
%kdTree = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');
%[ move{i}(:,4) ] = SparseGauss( kdTree, move{i}(:,4), 0.02);
  
     %move{i} = getNorms(move{i}, tform, 10000000);  
    
    tformMatB = createTformMat(tform);

    %get transformation matrix
    tformLady = ladybugParam.cam0.offset;
    tformMat = createTformMat(tformLady);
    tformMat = tformMat/tformMatB;
     
    m = filterScan(move{i}, metric, tformMat);

    LoadMoveScan(i-1,m,3);
    fprintf('loaded moving scan %i\n',i);
end

base = cell(numBase,1);
bStore = cell(numBase,1);
for i = 1:numBase
    idx2 = mod(i-1,5)+1;
    idx1 = (i - idx2)/5 + 1;
    baseIn = imread(basePaths{idx1,idx2});
    baseIn = imresize(baseIn,scale);
    mask = imread([path 'LadybugColourVideo\masks\cam' int2str(idx2-1) '.png']);
    mask = imresize(mask,scale);
    mask = mask(:,:,1);

    for q = 1:size(baseIn,3)
        temp = baseIn(:,:,q);
        temp(temp ~= 0) = histeq(temp(temp ~= 0));
        
        %G = fspecial('gaussian',[50 50],2);
        %temp = imfilter(temp,G,'same');
        
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
    
    if(strcmp(metric,'LIV'))
        if(i == 1)
            bAvg = b/numBase;
        else
            bAvg = bAvg + b/numBase;
        end
    end
    
    %base{i}.v = uint8(255*b(:,:,1));
    %LoadBaseImage(i-1,b);
    bStore{i} = b;
    fprintf('loaded base image %i\n',i);
end

for i = 1:size(bStore,1)
    LoadBaseImage(i-1,(bStore{i}));
end

%% get image alignment
tformTotal = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

for i = 1:numTrials
    [tformOut, fOut]=pso(@(tform) alignLadyVel2(base, move, pairs, tform, ladybugParam,scale), 6,[],[],[],[],param.lower,param.upper,[],param.options);

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