function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    image.v = histeq(image.v);
    out = single(image.v)/255;
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    [x,y] = gradient(out);
    mag = sqrt(x.^2 + y.^2);
    phase = 180*atan2(x,y)/pi;
    
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
else
    error('Invalid metric type');
end

end

