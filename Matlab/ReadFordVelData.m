function [data] = ReadFordVelData( pathIn, range )
%READVELDATA Reads velodyne data for ford dataset

folder = [pathIn 'scans/'];
files = dir(folder);
fileIndex = find(~[files.isdir]);

range = range(range <= length(fileIndex));

data = cell(length(range),1);
for i = 1:length(range)
    fileName = files(fileIndex(range(i))).name;
    
    data{i} = dlmread([folder fileName], ' ', 1, 0);
end



end

