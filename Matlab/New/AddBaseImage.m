function [] = AddBaseImage( image, tformIdx, scanIdx )
%ADDBASEIMAGE Adds a base image ready for calibration
% baseIn array holding image, dimensions order x,y,z (note needs coverting from matlabs y,x,z order).
% tformIdx index of the transform that will be applied to project moving scans onto this image
% scanIdx index of the scan that will be projected onto this image

height = size(image,1);
width = size(image,2);
depth = size(image,3);

image = single(image);
tformIdx = uint32(tformIdx(0));
scanIdx = uint32(scanIdx(0));

calllib('LibCal','addBaseImage',image, height, width, depth, tformIdx, scanIdx);

end

