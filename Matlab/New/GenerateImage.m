function [ image ] = GenerateImage(scanIdx, dilate, imageColour )
%GENERATEIMAGE Generates an image from the scan
% scanIdx- index of image attached to scan to use in generating image
% dilate- how much to dilate image by
% imageColour- true to use image colours, false to use scan colours


if(imageColour)
    depth = calllib('LibCal','getImageDepth',scanIdx);
else
    depth = calllib('LibCal','getNumCh',scanIdx);
end

width = calllib('LibCal','getImageWidth', scanIdx);
height = calllib('LibCal','getImageHeight', scanIdx);

imagePtr = libpointer('singlePtr',single(zeros(height,width,depth)));

calllib('LibCal','outputImage',imagePtr, width, height, scanIdx, dilate, imageColour);

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
    image(end,end,3) = 0;
end

end

