function [] = AddCamera( cIn, panoramic )
%ADDCAMERA  Adds a camera for use with cameraTransform, note may crash things if setup for other transform 
%cIn input camera array in coloum major form
% panoramic true if camera is panoramic, false otherwise

cIn = single(cIn);
panoramic = boolean(panoramic(0));
calllib('LibCal','addCamera',cIn, panoramic);

end

