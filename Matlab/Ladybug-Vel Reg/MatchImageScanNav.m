function [ basePaths, movePaths, pairs  ] = MatchImageScanNav( path, imRange, rep )
%MATCHIMAGESCAN Matches images to nearest scan with scans corrected by nav
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
        movePaths{j,1} = [path 'VelodyneLaser/scansNav/' fileName(1:16) '.bin'];
        
        pairs(j,:) = [j,j];
        %get time from name
        time(j) = str2double(fileName(1:16));
        j = j+1;
    end
   
    if(rep)
        %account for the 5 images per scan
        pairs = kron(pairs, ones(5,1));
        pairs(:,1) = 1:size(pairs,1);
    end
end

