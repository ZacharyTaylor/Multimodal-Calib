function [ basePaths, movePaths, pairs  ] = MatchFord( path, imRange, rep )
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

    basePaths = cell(length(imRange),5);
    movePaths = cell(length(imRange),1);
    pairs = zeros(length(imRange),2);
    
    folder = [path 'Ladybug/cam0/'];
    files = dir(folder);
    fileIndex = find(~[files.isdir]);

    j = 1;
    
    %for images in imRange
    for i = imRange
        %get name
        fileName = files(fileIndex(i)).name;
        fileName = char(fileName);
        
        %create base paths
        basePaths{j,1} = [path 'Ladybug/cam0/' fileName];
        basePaths{j,2} = [path 'Ladybug/cam1/' fileName];
        basePaths{j,3} = [path 'Ladybug/cam2/' fileName];
        basePaths{j,4} = [path 'Ladybug/cam3/' fileName];
        basePaths{j,5} = [path 'Ladybug/cam4/' fileName];

        j = j+1;
    end
    
    folder = [path 'Velodyne/Scans/'];
    files = dir(folder);
    fileIndex = find(~[files.isdir]);
    
    j = 1;
    
    %for images in imRange
    for i = imRange
        %get name
        fileName = files(fileIndex(i)).name;
        fileName = char(fileName);
        
        %create base paths
        movePaths{j,1} = [path 'Velodyne\Scans\' fileName];
        pairs(j,:) = [j, j];

        j = j+1;
    end

    %remove empty moving entries
    pairs = pairs(any(pairs,2),:);
    movePaths = movePaths(~cellfun('isempty',movePaths));  
    
    %find unique moving entries
    [~,idx,pairs(:,2)] = unique(char(movePaths),'rows');
    movePaths = movePaths(idx);
    
    %remove unused base entries
    basePaths = basePaths(pairs(:,1),:);
    pairs(:,1) = 1:length(pairs(:,1));
    
    if(rep)
        %account for the 5 images per scan
        pairs = kron(pairs, ones(5,1));
        pairs(:,1) = 1:size(pairs,1);
    end
end

