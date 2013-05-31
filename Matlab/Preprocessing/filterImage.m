function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    image.v = histeq(image.v);
    out = single(image.v)/255;
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    [mag,phase] = imgradient(out);
    
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
else
    error('Invalid metric type');
end

end

