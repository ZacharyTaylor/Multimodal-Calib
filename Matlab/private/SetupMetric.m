function [] = SetupMetric( metric )
%SETUPMETRIC Sets up specified metric for use

if(strcmp(metric, 'MI'))
    calllib('LibCal','setupMIMetric');
elseif(strcmp(metric, 'NMI'))
    calllib('LibCal','setupNMIMetric');
elseif(strcmp(metric, 'GOM'))
    calllib('LibCal','setupGOMMetric');
elseif(strcmp(metric, 'GOMS'))
    calllib('LibCal','setupGOMSMetric');
elseif(strcmp(metric, 'SSD'))
    SetupSSDMetric();
elseif(strcmp(metric, 'LEV'))
    calllib('LibCal','setupLEVMetric'); 
elseif(strcmp(metric, 'None'))
    
else
    error('Invalid metric');
end

end

