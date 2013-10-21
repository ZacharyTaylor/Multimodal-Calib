function [out] = getPointClouds(varargin)
%gets point cloud and converts them to a matrix
%call using [clouds] = getPointCloud(numClouds); or
%[clouds] = getPointCloud(numClouds,dlim);
% dlim is the dlimiter used when reading a csv ',' is default
% can be used to get any number of clouds at once
% can select multiple clouds an once

    pathName = 'C:\Users\ztay8963\Documents\GitHub\Multimodal-Calib\Sample Data\';
    if(nargin < 1)
        out{1} = [];
    else
        out{varargin{1},1} = [];
    end
    
    %get file names
    fileList = cell(size(out,1),1);
    pathList = cell(size(out,1),1);
    
    i = 1;
    while(i <= size(out,1))
        [fileName, pathName] = uigetfile(...
            {'*.csv;*.txt;*.mat','Point Cloud Files';...
            '*.*', 'All Files (*.*)'},...
            'Get Point Clouds',...
            pathName,...
            'MultiSelect', 'on');
        
        if(iscell(fileName))
            fileName = fileName';
            fileList(i:min(size(out,1),size(fileName,1)+i-1)) = fileName(1:min(size(out,1)-i+1,size(fileName,1)));
            pathList(i:min(size(out,1),size(fileName,1)+i-1)) = num2cell(pathName,2);
            
            i = i + size(fileName,1);
        else
            fileList{i} = fileName;
            pathList{i} = pathName;
            
            i = i+1;
        end
    end
    
    %get files
    for i = 1:size(out,1)

        if(isequal(strfind(fileList{i}, '.mat'),[]))
            if(nargin == 2)
                dlim = varargin{2};
            else
                dlim = ' ';
            end
            
            cloud = dlmread([pathList{i}, fileList{i}],dlim,0,0);
        else
            matIn = load([pathList{i}, fileList{i}]);
            names = fieldnames(matIn);
            
            if(size(names,1) > 1)
                cloud = [matIn.(names{1}), matIn.(names{2})];
            else
                cloud = matIn.(names{1});
            end
        
        end
    
        %center cloud
        %cloud(:,1:3) = bsxfun(@minus,cloud(:,1:3),(max(cloud(:,1:3)) + min(cloud(:,1:3)))/2);
        
        out{i} = cloud;
    end
end

