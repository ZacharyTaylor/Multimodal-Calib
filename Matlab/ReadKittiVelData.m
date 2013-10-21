function [data] = ReadKittiVelData( pathIn, range )
%READVELDATA Reads binary velodyne data

folder = [pathIn '\velodyne_points\data\'];
files = dir(folder);
fileIndex = find(~[files.isdir]);

range = range(range <= length(fileIndex));

data = cell(length(range),1);
for i = 1:length(range)
    fileName = files(fileIndex(range(i))).name;
    fid = fopen([folder fileName], 'r');
    
    data{i} = fread(fid,800000,'single');
    data{i} = reshape(data{i},4,[])';
    fclose(fid);
end



end

