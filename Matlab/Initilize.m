function [] = Initilize( numBase, numMove, numPairs )
%INITILIZE Sets up enviroment by loading library and setting up the
%required memory
%   numBase = number of Base images
%   numMove = number of moving scans
%   numPairs = number of pairs of base and moving images

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

if((numPairs ~= round(numPairs)) || (numPairs < 0))
    TRACE_ERROR('number of scan pairs must be a positive integer');
    numPairs = 0;
end

calllib('LibCal','initalizeScans', numBase, numMove, numPairs);

end

