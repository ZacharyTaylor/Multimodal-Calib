function [ solution ] = PCOptimize( levels, base, mask, metric, initalGuess, updatePeriod, dilate, varargin )
%OPTIMIZE Runs fminsearch to find optimum values

warning('off','images:initSize:adjustingMag');

solution = initalGuess;

for lev = levels:-1:1
    fprintf('Optimizing Level %i\n',lev);
    ChangePL( base, lev, metric, mask );
    
    if(nargin > 7)
        solution = fminsearch(@(tform) Align(tform, updatePeriod, dilate, varargin{1}), solution);
    else
        solution = fminsearch(@(tform) Align(tform, updatePeriod, dilate), solution);
    end
    
end

