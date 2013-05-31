function [ out ] = filterScan( scan, metric, tform)
%FILTERSCAN filters scan ready for use with metric

if(strcmp(metric,'MI'))
    out = single(scan);
    %out(:,4) = histeq(out(:,4));
    %[ ~,out(:,4) ] = getBetterNorms(out, 8, 100000);
    %out(isnan(out)) = 0;
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    
    [mag, phase ] = get2DGradient(out,tform);
    %[ mag, phase ] = getBetterNorms(out, tform, 1000000);
    
    %[mag,phase] = get2DGradient( out, tform );
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    %mag = histeq(mag);
    out = [out(:,1:3),mag,phase];
elseif(strcmp(metric,'LIV'))
    out = single(scan);
    out = livLidar(out);
    
else
    error('Invalid metric type');
end

%out(:,4) = out(:,4) - min(out(:,4));
out(:,4) = out(:,4) / max(out(:,4));
    
end

