function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    out = single(image.v)/255;
    out = histeq(out);
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    [mag,phase] = imgrad(out);
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    
    %mag(mag ~= 0) = histeq(mag(mag ~= 0));

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
elseif(strcmp(metric,'LIV'))
    out = single(image.v)/255;
    out = livImage(out);
    
else
    error('Invalid metric type');
end

end

