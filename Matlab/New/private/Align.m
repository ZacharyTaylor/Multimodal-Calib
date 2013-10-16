function [ out ] = Align( tform, updatePeriod, dilate )
%ALIGN Summary of this function goes here
%   Detailed explanation goes here

ClearTransforms();
AddTform(tform);

out = EvalMetric();

persistent time;
persistent fig;
if(isempty(time))
    time = clock;
end

if(isempty(fig))
    fig = figure;
end

%checks if figure exists and if it dosn't exits
if(~ishandle(fig))
    ClearEverything;
    clear fig;
    error('Program terminated by user')
end

if(etime(clock,time) > updatePeriod)
    
    numImages = calllib('LibCal','getNumImages');
    imNum = round((numImages-1)*rand(1));
    
    move = GenerateImage( imNum, dilate, false);
    move = move(:,:,1);

    base = GenerateImage( imNum, dilate, true);
    base = base(:,:,1);

    set(0,'CurrentFigure',fig)
    imshow([move;base]);
    drawnow;
    
    fprintf('Metric value = %1.3f ', out);
    fprintf('Current transform: [');
    for i = 1:length(tform)
        fprintf(' %1.3f', tform(i));
    end
    fprintf(']\n');
    
    time = clock;
end

out = -out;

end

