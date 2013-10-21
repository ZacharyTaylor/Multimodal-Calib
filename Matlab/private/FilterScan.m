function [ out ] = FilterScan( scan, metric, tform)
%FILTERSCAN filters scan ready for use with metric

if(or(strcmp(metric,'MI'),strcmp(metric,'NMI')))
    out = single(scan);
elseif(or(strcmp(metric,'GOM'),strcmp(metric,'GOMS')))   
    out = single(scan);
    
    [mag, phase ] = Get2DGradient(out,tform);
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    %mag(:) = histeq(mag(:));
    out = [out(:,1:3),mag,phase];
elseif(strcmp(metric,'LEV'))
    out = single(scan);
    out = livLidar(out);
elseif(strcmp(metric,'SSD'))
    out = single(scan);
elseif(strcmp(metric,'None'))
    out = single(scan);
else
    error('Invalid metric type');
end

out(:,4) = out(:,4) - min(out(:,4));
out(:,4) = out(:,4) / max(out(:,4));
    
end

