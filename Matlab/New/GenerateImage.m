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

image = single(zeros(width,height,depth));

imagePtr = calllib('LibCal','outputImage',image, width, height, scanIdx, dilate, imageColour);

setdatatype(imagePtr,'singlePtr',width,height*depth);
out = get(imagePtr);

end

