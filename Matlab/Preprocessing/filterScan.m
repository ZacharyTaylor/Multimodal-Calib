function [ out ] = filterScan( scan, metric )
%FILTERSCAN filters scan ready for use with metric

if(strcmp(metric,'MI'))
    out = single(scan);
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    [mag,phase] = getGradient(out);
    out = [out(:,1:3),mag,phase];
else
    error('Invalid metric type');
end
    
end

