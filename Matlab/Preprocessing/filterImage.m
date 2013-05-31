function [ out ] = filterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(strcmp(metric,'MI'))
    image.v = histeq(image.v);
    out = single(image.v)/255;
elseif(strcmp(metric,'GOM'))
    out = single(image.v)/255;
    [mag,phase] = imgradient(out);
    
    %G = fspecial('gaussian',[20 20],2);
    %mag = imfilter(mag,G,'same');
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    mag(mag ~= 0) = histeq(mag(mag ~= 0));

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
else
    error('Invalid metric type');
end

end

