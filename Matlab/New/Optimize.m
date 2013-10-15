function [ solution ] = Optimize( initalGuess, range, updatePeriod, dilate )
%OPTIMIZE Runs particle swarm to find optimum values

options = psooptimset('PopulationSize', 200,...
    'TolCon', 1e-1,...
    'StallGenLimit', 30,...
    'Generations', 200);

warning('off','images:initSize:adjustingMag');
    
lower = initalGuess - range;
upper = initalGuess + range;

solution =pso(@(tform) Align(tform, updatePeriod, dilate), length(initalGuess),[],[],[],[],lower,upper,[],options);

end

