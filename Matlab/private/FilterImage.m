function [ out ] = FilterImage( image, metric, mask )
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

image.v = MyHistEq(image.v);

if(or(strcmp(metric,'MI'),strcmp(metric,'NMI')))
    out = single(image.v);
    out(mask == 0) = 0;
elseif(strcmp(metric,'GOM'))
    out = single(image.v);
    
    %G = fspecial('gaussian',[50 50],1);
    %out = imfilter(out,G,'same');
    
    [x,y] = gradient(out);
    
    x(mask == 0) = 0;
    y(mask == 0) = 0;
    
    %G = fspecial('gaussian',[50 50],15);
  
    %x = imfilter(x,G,'same');
    %y = imfilter(y,G,'same');
    
    mag = sqrt(x.^2 + y.^2);
    phase = 180*atan2(x,y)/pi;
    %    phase = imfilter(phase,G,'same');
    
    %mag = mag - min(mag(:));
    %mag = mag / max(mag(:));
        
    mag = MyHistEq(mag);

    %mag = imfilter(mag,G,'same');
    
    out = zeros([size(mag) 2]);
    %out(:,:,1) = mag;
    

    out(:,:,1) = mag;%imfilter(mag,G,'same');
    out(:,:,2) = phase;%imfilter(phase,G,'same');
elseif(strcmp(metric,'GOMS'))
    out = single(image.v);
    
    G = fspecial('gaussian',[50 50],1);
    out = imfilter(out,G,'same');
    
    [x,y] = gradient(out);
    
    x(mask == 0) = 0;
    y(mask == 0) = 0;
       
    mag = sqrt(x.^2 + y.^2);
    phase = 180*atan2(x,y)/pi;
    phase = mod(phase+180,180);
    
    mag = mag - min(mag(:));
    mag = mag / max(mag(:));
        

    out = zeros([size(mag) 3]);
    
    out(:,:,1) = mag;%imfilter(mag,G,'same');
    out(:,:,2) = phase;%imfilter(phase,G,'same');
    out(:,:,3) = single(image.v);
    
elseif(strcmp(metric,'SSD'))
    out = single(image.v);
    out(mask == 0) = 0;
elseif(strcmp(metric,'LEV'))
    out = single(LevImage(image.v));
    out(mask == 0) = 0;
elseif(strcmp(metric,'None'))
    out = single(image.c);
    out(mask == 0) = 0;
else
    error('Invalid metric type');
end

end

