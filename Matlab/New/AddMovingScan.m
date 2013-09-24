function [] = AddMovingScan( scanL, scanI )
%ADDBASEIMAGE Adds a moving scan ready for calibration
% baseIn array holding image, dimensions order x,y,z (note needs coverting from matlabs y,x,z order).
% tformIdx index of the transform that will be applied to project moving scans onto this image
% scanIdx index of the scan that will be projected onto this image

scanL = single(scanL);
scanI = single(scanI);

length = min(size(scanL,1),size(scanI,1));
numDim = size(scanL,2);
numCh = size(scanI,2);

calllib('LibCal','addMovingScan', scanL, scanI, length, numDim, numCh);

end

