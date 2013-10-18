function [] = SetupMetric( metric )
%SETUPMETRIC Sets up specified metric for use

if(strcmp(metric, 'NMI'))
    SetupNMIMetric();
elseif(strcmp(metric, 'GOM'))
    SetupGOMMetric();
elseif(strcmp(metric, 'SSD'))
    SetupSSDMetric();
elseif(strcmp(metric, 'LIV'))
    printf('Not written yet\n');   
elseif(strcmp(metric, 'None'))
    
else
    printf('Invalid metric\n');
end

end

