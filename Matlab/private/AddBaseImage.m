function [] = AddBaseImage( image )
%ADDBASEIMAGE Adds a base image ready for calibration
% baseIn array holding image, dimensions order y,x,z.

height = size(image,1);
width = size(image,2);
depth = size(image,3);

image = single(image);

calllib('LibCal','addBaseImage',image, height, width, depth);

end

