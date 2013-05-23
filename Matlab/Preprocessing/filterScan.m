function [ out ] = filterScan( scan, metric, tform)
%FILTERSCAN filters scan ready for use with metric

if(strcmp(metric,'MI'))
    out = single(scan);
    %[ ~,out(:,4) ] = getBetterNorms(out, tform, 100000);
    out(isnan(out)) = 0;
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    [ mag, phase ] = getGradient( out, tform );
    %[ mag, phase ] = getBetterNorms(out, tform, 1000000);
    
    %[mag,phase] = get2DGradient( out, tform );
    mag(mag ~= 0) = histeq(mag(mag ~= 0));
    out = [out(:,1:3),mag,phase];
else
    error('Invalid metric type');
end

out(:,4) = out(:,4) - min(out(:,4));
out(:,4) = out(:,4) / max(out(:,4));
    
end

