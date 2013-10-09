function [] = SetupMetric( metric )
%SETUPMETRIC Sets up specified metric for use

if(strcmp(metric, 'NMI'))
    printf('Not written yet\n');
elseif(strcmp(metric, 'GOM'))
    SetupGOMMetric();
elseif(strcmp(metric, 'SSD'))
    SetupSSDMetric();
elseif(strcmp(metric, 'LIV'))
    printf('Not written yet\n');   
else
    printf('Invalid metric\n');
end

end

