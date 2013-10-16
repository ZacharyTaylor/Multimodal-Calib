function [] = ClearEverything()
%CLEAREVERYTHING Clears all allocated memory

calllib('LibCal','clearEverything');
unloadlibrary('LibCal');

end

