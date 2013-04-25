function [ out ] = filterScan( scan, metric, tform)
%FILTERSCAN filters scan ready for use with metric

if(strcmp(metric,'MI'))
    out = single(scan);
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    [mag,phase] = get2DGradient( out, tform );
    out = [out(:,1:3),mag,phase];
else
    error('Invalid metric type');
end
    
end

