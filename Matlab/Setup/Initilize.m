function [] = Initilize( numBase, numMove)
%INITILIZE Sets up enviroment by loading library and setting up the
%required memory
%   numBase = number of Base images
%   numMove = number of moving scans

%load the library
CheckLoaded();

%check inputs
if((numBase ~= round(numBase)) || (numBase < 0))
    TRACE_ERROR('number of base scans must be a positive integer');
    numBase = 0;
end

if((numMove ~= round(numMove)) || (numMove < 0))
    TRACE_ERROR('number of move scans must be a positive integer');
    numMove = 0;
end

calllib('LibCal','initalizeScans', numBase, numMove, 1);

end

