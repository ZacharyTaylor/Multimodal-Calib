function [ out ] = Align( tform, updatePeriod, dilate )
%ALIGN Summary of this function goes here
%   Detailed explanation goes here

if(length(tform) >= 9)
    camera = tform(1,7:end);
    tform = tform(1,1:6);
    
    pan = calllib('LibCal','getIfPanoramic',0);
    ClearCameras();
    AddCamera(camera,pan);
end

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
    
    subplot(2,3,4:6);
    numImages = calllib('LibCal','getNumImages');
    imNum = round((numImages-1)*rand(1));
    
    move = GenerateImage( imNum, dilate, false);
    move = repmat(move(:,:,1),[1,1,3]);

    base = GenerateImage( imNum, dilate, true);
    base = repmat(base(:,:,1),[1,1,3]);

    C = imfuse(move(:,:,1),base(:,:,1),'falsecolor','Scaling','independent','ColorChannels',[2 1 2]);
    
    set(0,'CurrentFigure',fig)
    imshow([move;base;C]);
    drawnow;
    
    fprintf('Metric value = %1.3f ', out);
    fprintf('Current transform: [');
    for i = 1:length(tform)
        fprintf(' %1.3f', tform(i));
    end
    fprintf(']\n');
    
    time = clock;
end

end

