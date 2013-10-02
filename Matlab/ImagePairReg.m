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
FIG.countMax = 1000;

%range to search over (x, y, r, sx, sy, kx, ky)
range = [30 30 0 0 0 0 0];

%inital guess of parameters (x, y, r, sx, sy, kx, ky)
tform = [0 0 0 1 1 0 0];

%number of images
numMove = 1;
numBase = 1;

%pairing [base image, move image]
pairs = [1 1];

%number of times to run
numTrials = 1;

%metric to use
metric = 'GOM';

%% setup transform and images
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

%% get Data
move = getImagesC(numMove);

for i = 1:numMove
    m = filterImage(move{i}, metric);
    LoadMoveImage(i-1,m);
end

base = getImagesC(numBase);

for i = 1:numBase
    b = filterImage(base{i}, metric);   
    LoadBaseImage(i-1,b);
end



%% get image alignment

tformTotal = zeros(numTrials,size(tform,2));
tformIn = zeros(numTrials,size(tform,2));
fTotal = zeros(numTrials,1);

i = 0;
val = zeros(101,1);
for x = -0.5:0.01:0.5
    i = i+1;
        val(i) = alignImages(base, move, pairs, [0,x*320,0,1,1,0,0]);
end
plot(val(:,1));
% for i = 1:numTrials
%     
%     tformIn(i,:) = tform + (2*rand(size(tform,1), size(tform,2)) - 1).*range;
%     tformRun = tformIn(i,:);
%     
%     [tformOut, fOut]=pso(@(tformRun) alignImages(base, move, pairs, tformRun), 7,[],[],[],[],param.lower,param.upper,[],param.options);
% 
%     tformTotal(i,:) = tformOut;
%     fTotal(i) = fOut;
% end

% tform = sum(tformTotal,1) / numTrials;
% f = sum(fTotal) / numTrials;
% 
%  fprintf('Final transform:\n     metric = %1.3f\n     translation = [%3.0f, %3.0f]\n     rotation = %1.2f\n     scale = [%1.2f,%1.2f]\n     shear = [%0.3f, %0.3f]\n\n',...
%             f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6),tform(7));

%% cleanup
ClearLibrary;
rmPaths;