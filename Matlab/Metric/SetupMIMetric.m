function [] = SetupMIMetric()
%SETUPMIMETRIC sets up the mutual information metric

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setupMIMetric',50);

end