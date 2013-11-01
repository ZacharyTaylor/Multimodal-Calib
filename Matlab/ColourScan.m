function [ scan ] = ColourScan(scanIdx)
%GENERATEIMAGE Generates a scan coloured by indexed image
% scanIdx- index of image attached to scan

numPoints = calllib('LibCal','getNumPoints',scanIdx);
dimensions = calllib('LibCal','getNumDim',scanIdx);
channels = calllib('LibCal','getImageDepth',scanIdx);

scanPtr = libpointer('singlePtr',single(zeros(numPoints,(dimensions + channels))));

calllib('LibCal','colourScan',scanPtr, scanIdx);

scan = get(scanPtr,'value');
scan = reshape(scan,numPoints,(dimensions + channels));

scan = scan(any(scan(:,dimensions+1:end),2),:);

%scan(:,dimensions+1:end) = round(255*scan(:,dimensions+1:end));

end

