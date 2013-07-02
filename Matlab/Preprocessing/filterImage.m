function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

image.v = single(image.v);
image.v(:) = image.v(:) - min(image.v(:));
image.v(:) = image.v(:) / max(image.v(:));
image.v(:) = histeq(image.v(:));

if(strcmp(metric,'MI'))
    out = image.v;
elseif(strcmp(metric,'GOM'))
    out = image.v;
    %[mag,phase] = getgrad2Im(out);
    [mag,phase] = imgradient(out);
    
    %G = fspecial('gaussian',[20 20],2);
    %mag = imfilter(mag,G,'same');
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    %mag(mag ~= 0) = histeq(mag(mag ~= 0));

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
elseif(strcmp(metric,'LIV'))
    out = image.v;
    out = livImage(out);
else
    error('Invalid metric type');
end

end

