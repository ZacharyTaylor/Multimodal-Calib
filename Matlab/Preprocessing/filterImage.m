function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    out = single(image.v)/255;
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    [mag,phase] = imgradient(out);
    
    phase = abs(phase);
    mag = histeq(mag);

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
else
    error('Invalid metric type');
end

end

