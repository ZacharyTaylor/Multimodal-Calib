function [] = AddTform( tform )
%ADDTFORM Adds a transform that will be used to project moving scans onto base images
% tformIn array holding transform

tform = single(tform(:,:,1));
tformSizeX = size(tform,2);
tformSizeY = size(tform,1);

calllib('LibCal','addTform',tform,tformSizeX,tformSizeY);

end

