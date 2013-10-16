function [] = Setup( metric, move, base, tform, varargin )
%SETUP Summary of this function goes here
%   metric- metric to use for comparing images
%           GOM gradient orientation measure
%           SSD sum of squared differences
%           NMI normalized mutual information
%           NONE no metric (for use with colouring scans)
%   move- cell containing moving scans or images
%   base- cell of base images
%   tform- cell of inital transforms
%   cam- (optional) inital camera
%   panFlag- (optional) if true uses panoramic camera, else pin-point(default) 
    

%% Setup library
SetupLib();

%% Check, format and filter inputs
%check for a camera input
if(nargin > 0)
    cam = varargin{1};
end
    
scan = exist('cam','var');
if(scan)
    %check for a camera panoramic flag
    if(nargin > 1)
        panFlag = varargin{2};
    end
end

%convert to cells
temp = cell(1,1);
if(~iscell(move))
    temp{1} = move;
    move = temp;
end
if(~iscell(base))
    temp{1} = base;
    base = temp;
end
if(~iscell(tform))
    temp{1} = tform;
    tform = temp;
end
if(exist('cam','var') && ~iscell(cam))
    temp{1} = cam;
    cam = temp;
end

%check sizes
if(size(move,2) > 1)
    error('Error move must be n by 1 in size');
end
if(size(move,1) ~= size(base,1))
    error('Error move and base require same number of columns')
end
if(((size(tform,1) ~= 1) && (size(tform,1) ~= size(base,1))) || ((size(tform,2) ~= 1) && (size(tform,2) ~= size(base,2))))
    error('Error if base is size n by m then possible tform sizes are 1 by 1, 1 by m, n by 1 or n by m');
end
if(exist('cam','var') && (((size(cam,1) ~= 1) && (size(cam,1) ~= size(base,1))) || ((size(cam,2) ~= 1) && (size(cam,2) ~= size(base,2)))))
    error('Error if base is size n by m then possible cam sizes are 1 by 1, 1 by m, n by 1 or n by m');
end

%filter moving scans
for i = 1:size(move(:),1)
    if(scan)
        move{i} = FilterScan(move{i}, metric, tform{1});
    else
        move{i} = FilterImage(move{i}, metric);
    end
end

%filter base images
for i = 1:size(base(:),1)
    base{i} = FilterImage(base{i}, metric);
end

%% Initalize GPU
Initilize(scan);

%% Clear old stuff
ClearCameras();
ClearImages();
ClearIndices();
ClearScans();
ClearTransforms();

%% Setup metric
SetupMetric(metric);

%% Transfer to GPU
%load scans
for i = 1:size(move(:),1)
    if(scan)
        AddMovingScan(move{i}(:,1:3), move{i}(:,4:end));
    else
        AddMovingImage(move{i});
    end
end
%load images
for i = 1:size(base(:),1)
    AddBaseImage(base{i});
end
%load tforms
for i = 1:size(tform(:),1)
    AddTform(tform{i});
end
%load cameras
if(exist('cam','var'))
    if(~exist('panFlag','var'))
        panFlag = false;
    end
    for i = 1:size(cam(:),1)
        AddCamera(cam{i}, panFlag);
    end
end

%% Setup indicies
if(exist('cam','var'))
    SetupIndex(size(move), size(base), size(tform), size(cam));
else
    SetupIndex(size(move), size(base), size(tform), 0);
end

end

