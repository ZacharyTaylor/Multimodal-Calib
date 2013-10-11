function [] = Setup( metric, move, base, tform, varargin )
%SETUP Summary of this function goes here
%   Detailed explanation goes here

%% Setup library
SetupLib();

%% Check, format and filter inputs

%check for a camera input
if(nargin > 0)
    cam = varargin{1};
end
%check for a camera panoramic flag
if(nargin > 1)
    panFlag = varargin{2};
end
%check for a base transform
if(nargin > 2)
    baseTform = varargin{3};
end

%convert to cells
if(~iscell(move))
    move{1} = move;
end
if(~iscell(base))
    base{1} = base;
end
if(~iscell(tform))
    tform{1} = tform;
end
if(exist('cam','var') && ~iscell(cam))
    cam{1} = cam;
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
    if(exist('baseTform','var'))
        move{i} = FilterScan(move{i}, metric, baseTform);
    elseif(size(tform,2) == 1)
        move{i} = FilterScan(move{i}, metric, tform{i});
    else
        error('Error setups that require multiple transforms per scan require a base transform');
    end
end

%filter base images
for i = 1:size(base(:),1)
    base{i} = FilterImage(base{i}, metric);
end

%% Initalize GPU
Initilize(exist('cam','var'));

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
    AddMovingScan(move{i}(:,1:3), move{i}(:,4:end));
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

