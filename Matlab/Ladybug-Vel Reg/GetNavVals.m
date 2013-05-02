function [ pos  ] = GetNavVals( path, basePaths )
%MATCHIMAGESCAN Matches images to nearest scan
%   input 
%   imRange- range of images to get matches for
%   path- base path to begin looking in
%   rep- repeat for each of the cameras
%   
%   output
%   basePaths - paths to base images
%   movePaths - paths to moving scans
%   pairs - matched pairs of base images and moving scans index

    basePaths = basePaths(:,1);
    
    %load nav
    nav = dlmread([path 'NovatelNav\position.csv'],',');
    
    %get image times
    basePaths = char(basePaths);
    times = str2num(basePaths(:,(end-21):(end-9)));
    
    pos = zeros(size(times,1),6);
    
    for i = 1:size(times)
        tmp = abs(times(i)-nav(:,1));
        [~, idx] = min(tmp);
        pos(i,:) = nav(idx(1),2:end);
    end
end

