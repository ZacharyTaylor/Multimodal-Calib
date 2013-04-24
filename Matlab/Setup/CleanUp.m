function [] = CleanUp()
%CLEANUP Summary of this function goes here
%   Detailed explanation goes here

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','clearScans');
calllib('LibCal','clearMetric');
calllib('LibCal','clearTform');

end

