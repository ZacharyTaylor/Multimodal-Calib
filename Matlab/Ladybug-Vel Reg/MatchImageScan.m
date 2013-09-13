function [ basePaths, movePaths, pairs  ] = MatchImageScan( path, imRange, rep )
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
    
    folder = [path 'LadybugColourVideo/cam0/'];
    files = dir(folder);
    fileIndex = find(~[files.isdir]);

    time = zeros(length(imRange),1);
    j = 1;
    
    %for images in imRange
    for i = imRange
        %get name
        fileName = files(fileIndex(i)).name;
        fileName = char(fileName);
        
        %create base paths
        basePaths{j,1} = [path 'LadybugColourVideo/cam0/' fileName(1:16) '.cam0.png'];
        basePaths{j,2} = [path 'LadybugColourVideo/cam1/' fileName(1:16) '.cam1.png'];
        basePaths{j,3} = [path 'LadybugColourVideo/cam2/' fileName(1:16) '.cam2.png'];
        basePaths{j,4} = [path 'LadybugColourVideo/cam3/' fileName(1:16) '.cam3.png'];
        basePaths{j,5} = [path 'LadybugColourVideo/cam4/' fileName(1:16) '.cam4.png'];

        %get time from name
        time(j) = str2double(fileName(1:16));
        j = j+1;
    end

    folder = [path 'VelodyneLaser/Scans/'];
    files = dir(folder);
    fileIndex = find(~[files.isdir]);
   
    fileName = cell(length(fileIndex),1);
    out = cell(length(fileIndex),3);
    for i = 1:length(fileIndex)
        fileName{i} = files(fileIndex(i)).name;
        out(i,:) = textscan(char(fileName{i}), 'Scan %d Time %f_%f.csv');
    end
    timeMin = cell2mat(out(:,2));
    timeMax = cell2mat(out(:,3));
           
    %for images in imRange
    j = 1;
    for i = imRange

        %find the matching image
        match = (timeMin < time(j)) & (timeMax > time(j));
        [V, idx] = max(match);
        if(V ~= 0)
            pairs(j,:) = [j, j];
            movePaths{j,1} = [folder fileName{idx(1)}];
        end
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

