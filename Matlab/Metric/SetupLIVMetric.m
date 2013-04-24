function [] = SetupLIVMetric()
%SETUPLIVMETRIC sets up the mutual information metric

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setupLIVMetric');

end