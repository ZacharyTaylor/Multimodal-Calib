function [ out ] = FilterImage( image, metric )
%FILTERIMAGE filters image ready for use with metric

if(~isstruct(image))
    temp = image;
    image = struct;
    if(size(temp,3) == 3)
        image.v = rgb2gray(temp);
    else
        image.v = temp(:,:,1);
    end
    
    image.c = temp;
end

if(or(strcmp(metric,'MI'),strcmp(metric,'NMI')))
    image.v = histeq(image.v);
    out = single(image.v)/255;
elseif(or(strcmp(metric,'GOM'),strcmp(metric,'GOMS')))
    out = single(image.v)/255;
    [x,y] = gradient(out);
    mag = sqrt(x.^2 + y.^2);
    phase = 180*atan2(x,y)/pi;
    
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
    
    mag = histeq(mag);

    out = zeros([size(mag) 2]);
    out(:,:,1) = mag;
    out(:,:,2) = phase;
elseif(strcmp(metric,'SSD'))
    image.v = histeq(image.v);
    out = single(image.v)/255;
elseif(strcmp(metric,'None'))
    out = single(image.c)/255;
else
    error('Invalid metric type');
end

end

