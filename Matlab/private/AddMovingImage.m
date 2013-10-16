function [] = AddMovingImage( Image )
%ADDBASEIMAGE Adds a moving image ready for calibration

x = repmat(1:size(Image,2),size(Image,1),1) - 1;
y = repmat((1:size(Image,1))',1,size(Image,2)) - 1;

scanI = reshape(Image,size(Image,1)*size(Image,2),[]);
scanL = [x(:), y(:)];

scanL = single(scanL);
scanI = single(scanI);

length = min(size(scanL,1),size(scanI,1));
numDim = size(scanL,2);
numCh = size(scanI,2);

calllib('LibCal','addMovingScan', scanL, scanI, length, numDim, numCh);

end

