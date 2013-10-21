function [] = ClearEverything()
%CLEAREVERYTHING Clears all allocated memory

clearvars -global fig

calllib('LibCal','clearEverything');
unloadlibrary('LibCal');

end

