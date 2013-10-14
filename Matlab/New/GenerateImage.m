function [ image ] = GenerateImage( width, height, scanIdx, dilate, imageColour )
%GENERATEIMAGE Generates an image from the scan
% width- width of image
% height- height of image
% scanIdx- index of image attached to scan to use in generating image
% dilate- how much to dilate image by
% imageColour- true to use image colours, false to use scan colours


if(imageColour)
    depth = calllib('LibCal','getImageDepth',scanIdx);
else
    depth = calllib('LibCal','getNumCh',scanIdx);
end

imagePtr = libpointer('singlePtr',single(zeros(height,width,depth)));

calllib('LibCal','outputImage',imagePtr, width, height, scanIdx, dilate, imageColour);

image = get(imagePtr,'value');

image = image - min(image(:));
image = image/max(image(:));
image(image ~= 0) = histeq(image(image ~= 0));

end

