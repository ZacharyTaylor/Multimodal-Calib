function [] = SetupSSDMetric()
%SETUPSSDMETRIC Sets up SSD metric ready for use
%metric is normalized to prevent NaN errors that are pretty common
calllib('LibCal','setupSSDMetric');
end

