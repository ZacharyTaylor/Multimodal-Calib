function [ image ] = GetBaseImage(baseIdx)
%GENERATEIMAGE Gets an image
% scanIdx- index of image

depth = calllib('LibCal','getImageDepth',baseIdx);
width = calllib('LibCal','getImageWidth', baseIdx);
height = calllib('LibCal','getImageHeight', baseIdx);

imagePtr = libpointer('singlePtr',single(zeros(width,height,depth)));

calllib('LibCal','outputBaseImage',imagePtr, baseIdx);

image = get(imagePtr,'value');
image = reshape(image,height,width,depth);

for i = 1:depth
    temp = image(:,:,i);
    temp = temp - min(temp(:));
    temp = temp/max(temp(:));
    temp(temp ~= 0) = histeq(temp(temp ~= 0));
    
    image(:,:,i) = temp;
end

if(depth == 2)
    image = image(:,:,1);
end

image = uint8(255*image);

end

