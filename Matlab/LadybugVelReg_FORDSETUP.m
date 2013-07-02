%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','normal');
%clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

global FIG
FIG.fig = figure;
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure. 
FIG.count = 0;

%get ladybug parameters
ladybugParam = FordConfig;

%% input values
param = struct;

%options for swarm optimization
param.options = psooptimset('PopulationSize', 200,...
    'TolCon', 1e-10,...
    'StallGenLimit', 50,...
    'Generations', 100,...
    'PlotFcns',{@psoplotbestf,@psoplotswarmsurf});

%how often to display an output frame
FIG.countMax = 5000;

scale = 0.25;

%range to search over (x, y ,z, rX, rY, rZ)
range = [0.5 0.5 0.5 3 3 3];
range(4:6) = pi*range(4:6)/180;

%inital guess of parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'G:\DataSets\Mobile Sensor Plaforms\Ford\mi-extrinsic-calib-data\';
%range of images to use
imRange = 1:20;%sort(1+ round(19*rand(5,1)))'
%metric to use
metric = 'GOM';

%number of times to run optimization
numTrials = 10;

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

[basePaths, movePaths, pairs] = MatchFord( path, imRange, true);

numBase = max(pairs(:,1));
numMove = max(pairs(:,2));

Initilize(numBase, numMove);

%% get move{i}
move = cell(numMove,1);
for i = 1:numMove
    move{i} = dlmread(movePaths{i},' ',1,0);
    %move{i} = ReadVelData(movePaths{i});

    %move{i}(:,4) = sqrt(move{i}(:,1).^2 + move{i}(:,2).^2 + move{i}(:,3).^2);
    %move{i} = getNorms(move{i}, tform);
    
    t = [0 0 0 0 0 0];
    tformMat = createTformMat(t);
    m = filterScan(move{i}, metric, tformMat);
    
    LoadMoveScan(i-1,m,3);
    fprintf('loaded moving scan %i\n',i);
end
tformTotal = zeros(numTrials,2*size(tform,2),5);
fTotal = zeros(numTrials,5);

savename = 'GOM_5';
    for j = 1:5
base = cell(numBase,1);
bStore = cell(numBase,1);
for i = j:5:numBase
    idx2 = mod(i-1,5)+1;
    idx1 = (i - idx2)/5 + 1;
    baseIn = imread(basePaths{idx1,idx2});
    baseIn = imresize(baseIn,scale);
    mask = imread([path 'Ladybug\masks\cam' int2str(idx2-1) '.png']);
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
        if(i == j)
            bAvg = 5*b/numBase;
        else
            bAvg = bAvg + 5*b/numBase;
        end
    end
    
    
    %base{i}.v = uint8(255*b(:,:,1));
    bStore{i} = b;
    %LoadBaseImage(i-1,b);
    fprintf('loaded base image %i\n',i);
end

for i = j:5:size(bStore,1)
    if(strcmp(metric,'LIV'))
        LoadBaseImage(i-1,(bStore{i}-bAvg));
    else
        LoadBaseImage(i-1,(bStore{i}));
    end
end

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

%% get image alignment


for i = 1:numTrials

        tformIn = tform + (rand(1,6)-1).*range;

        param.lower = tformIn - range;
        param.upper = tformIn + range;

        [tformOut, fOut]=pso(@(tformIn) alignLadyVel(base, move, pairs, tformIn, ladybugParam, j,scale), 6,[],[],[],[],param.lower,param.upper,[],param.options);

        tformTotal(i,1:6,j) = tformIn;
        tformTotal(i,7:12,j) = tformOut;
        fTotal(i,j) = fOut;
        
        i
        save([savename '_tform'],'tformTotal');
        save([savename '_f'],'fTotal');
    end
    j
end

fprintf('Final transform:\n     metric = %1.3f\n     rotation = [%2.2f, %2.2f, %2.2f]\n     translation = [%2.2f, %2.2f, %2.2f]\n\n',...
            f,tform(4),tform(5),tform(6),tform(1),tform(2),tform(3));
        
%% cleanup
ClearLibrary;
rmPaths;