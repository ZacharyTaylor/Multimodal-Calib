function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    out = single(image.v)/255;
    out = histeq(out);
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    out = histeq(out);
    [mag,phase] = imgradient(out);

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
else
    error('Invalid metric type');
end

end

