function [ out ] = Align( tform, updatePeriod, dilate, varargin )
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

if(nargin > 3)
    multicam = varargin{1};
    for i = 1:length(multicam(:))
        tb = CreateTformMat(tform);
        tc = CreateTformMat(multicam{1,i});
        AddTform(tc\tb);
    end
else
    AddTform(tform);
end

out = EvalMetric();

persistent time;
global fig;
if(isempty(time))
    time = clock;
end

if(isempty(fig))
    fig = figure;
end

%checks if figure exists and if it dosn't exits
if(~ishandle(fig))
    fig = figure;
%     ClearEverything;
%     clear fig;
%     error('Program terminated by user')
end

if(etime(clock,time) > updatePeriod)
    
    subplot(2,3,4:6);
    numImages = calllib('LibCal','getNumImages');
    imNum = round((numImages-1)*rand(1));
    %imNum = 0;
 
    move = GenerateImage( imNum, dilate, false);   
    move = repmat(move(:,:,1),[1,1,3]);

    base = GetBaseImage(imNum);
    if(size(base,3) == 1)
        base = repmat(base(:,:,1),[1,1,3]);
    end

    C = imfuse(move(:,:,1),base(:,:,1),'falsecolor','Scaling','independent','ColorChannels',[2 1 2]);
    
    set(0,'CurrentFigure',fig)
    if(size(move,1) > size(move,2))
        imshow([move,base,C]);
    else
        imshow([move;base;C]);
    end
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

