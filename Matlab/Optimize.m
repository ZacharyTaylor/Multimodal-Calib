function [ solution ] = Optimize( initalGuess, range, updatePeriod, dilate, varargin )
%OPTIMIZE Runs particle swarm to find optimum values

addpath('./psopt');

options = psooptimset('PopulationSize', 500,...
    'TolCon', 1e-1,...
    'StallGenLimit', 50,...
    ...%'PlotFcns',{@AlignPlotSwarm},...
    'Generations', 200);

warning('off','images:initSize:adjustingMag');

lower = initalGuess - range;
upper = initalGuess + range;

if(nargin > 4)
    solution =pso(@(tform) Align(tform, updatePeriod, dilate, varargin{1}), length(initalGuess),[],[],[],[],lower,upper,[],options);
else
    solution =pso(@(tform) Align(tform, updatePeriod, dilate), length(initalGuess),[],[],[],[],lower,upper,[],options);
end
end

