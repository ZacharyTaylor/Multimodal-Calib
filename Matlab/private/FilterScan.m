function [ out ] = FilterScan( scan, metric, tform)
%FILTERSCAN filters scan ready for use with metric

scan = double(scan);
%scan(:,4) = scan(:,4) - min(scan(:,4));
%scan(:,4) = scan(:,4) / max(scan(:,4));
scan(:,4) = MyHistEq(scan(:,4));

if(or(strcmp(metric,'MI'),strcmp(metric,'NMI')))
    out = single(scan);
elseif(strcmp(metric,'GOM'))   
    out = single(scan);
    
    [x,y] = Get2DGradProject(out,tform);
    
    mag = sqrt(x.^2 + y.^2);
    phase = 180*atan2(x,y)/pi;
    
    mag(:) = MyHistEq(mag(:));
    %phase = phase(mag < 0.8);
    %out = out(mag < 0.8,1:3);
    %mag = mag(mag < 0.8);
    
    out = [out(:,1:3),mag,phase];
elseif(strcmp(metric,'GOMS'))
    out = single(scan);
    
    [mag, phase ] = Get2DGradient(out,tform);
    
    out = [out(:,1:3),mag,phase, out(:,4)];
    
elseif(strcmp(metric,'LEV'))
    out = single(scan);
    out = LevLidar(out);
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

