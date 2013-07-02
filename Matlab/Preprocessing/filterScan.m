function [ out ] = filterScan( scan, metric, tformMat)
%FILTERSCAN filters scan ready for use with metric

scan = single(scan);
scan(:,4) = scan(:,4) - min(scan(:,4));
scan(:,4) = scan(:,4) / max(scan(:,4));
scan(:,4) = histeq(scan(:,4));

if(strcmp(metric,'MI'))
    out = single(scan);
    %[ ~,out(:,4) ] = getBetterNorms(out, tform, 100000);
    out(isnan(out)) = 0;
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    [ mag, phase ] = getGradient( out, tformMat );
    
    %temp = histeq(mag);
   
    
    out = [out(:,1:3),mag,phase];
    
    %thin
    %out = out(temp > 0.90,:);
elseif(strcmp(metric,'LIV'))
    out = single(scan);
   
    out = livLidar(out);
    
else
    error('Invalid metric type');
end

out(:,4) = out(:,4) - min(out(:,4));
out(:,4) = out(:,4) / max(out(:,4));
    
end

