function [] = SetupGOMMetric()
%SETUPGOMMETRIC sets up the mutual information metric

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setupGOMMetric');

end