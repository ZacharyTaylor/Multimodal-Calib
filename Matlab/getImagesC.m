function [out] = getImagesC(varargin)
%gets numOut number of images and returns cell of them where each member isloaded into a struct with r,g,b and v (grey) channels
%call using [out] = getImagesC(); or
%           [out] = getImagesC(numOut); or
%           [out] = getImagesC(numOut, histEq);
% can be used to get any number of images at once
% can select multiple images an once

    pathName = 'C:\Users\ztay8963\Documents\GitHub\Multimodal-Calib\Sample Data\';
    if(nargin < 1)
        out{1} = [];
    else
        out{varargin{1},1} = [];
    end
    
    if(nargin > 1)
        histEq = varargin{2};
    else
        histEq = false;
    end
    
    %get file names
    fileList = cell(size(out,1),1);
    pathList = cell(size(out,1),1);
    
    i = 1;
    while(i <= size(out,1))
        [fileName, pathName] = uigetfile(...
            {'*.jpg;*.png;*.jpeg;*.gif;*.tiff;*.tif;*.bmp;*.ppm;*.mat','Image Files';...
            '*.*', 'All Files (*.*)'},...
            'Get Images',...
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
            imgIn = imread([pathList{i}, fileList{i}]);

            img = struct;
            
            if(histEq)
                for j = 1:size(imgIn,3)
                    temp = imgIn(:,:,j);
                    temp(temp ~= 0) = histeq(temp(temp ~= 0));
                    imgIn(:,:,j) = temp;
                end
            end
                       
            if(size(imgIn,3)==3)
                img.c = imgIn;
                img.v = rgb2gray(imgIn);
            else
                img.c = imgIn(:,:,1);
                img.v = imgIn(:,:,1);
            end

        else
            img = load([pathList{i}, fileList{i}]);
            names = fieldnames(img);
            img = img.(names{1});
        end

        out{i} = img;
    end
end

