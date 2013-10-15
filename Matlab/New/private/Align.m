function [ out ] = Align( tform, updatePeriod, dilate )
%ALIGN Summary of this function goes here
%   Detailed explanation goes here

ClearTransforms();
AddTform(tform);
out = EvalMetric();

persistent time;
if(isempty(time))
    time = clock;
end

if(etime(clock,time) > updatePeriod)
    
    numImages = calllib('LibCal','getNumImages');
    imNum = round((numImages-1)*rand(1));
    
    move = GenerateImage( imNum, dilate, false);
    move = move(:,:,1);

    base = GenerateImage( imNum, dilate, true);
    base = base(:,:,1);

    imshow([move;base]);
    drawnow;
end


end

