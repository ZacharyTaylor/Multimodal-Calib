function [ solution ] = ConvexOptimize( initalGuess, updatePeriod, dilate, varargin )
%OPTIMIZE Runs fminsearch to find optimum values

warning('off','images:initSize:adjustingMag');

if(nargin > 3)
    solution = fminsearch(@(tform) Align(tform, updatePeriod, dilate, varargin{1}), initalGuess);
else
    solution = fminsearch(@(tform) Align(tform, updatePeriod, dilate), initalGuess);
end
end

