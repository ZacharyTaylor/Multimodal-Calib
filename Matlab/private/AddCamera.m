function [] = AddCamera( cIn, panoramic )
%ADDCAMERA  Adds a camera for use with cameraTransform, note may crash things if setup for other transform 
%cIn input camera array in coloum major form
% panoramic true if camera is panoramic, false otherwise
%
%If cIn is of size (1,3) will assume in form [f,ccX,ccY]
%If cIn is of size (1,4) will assume in form [fX,fY,ccX,ccY]

if(isequal(size(cIn),[1,4]))
    cIn = [cIn(1) 0 cIn(3) 0; 0 cIn(2) cIn(4) 0; 0 0 1 0];
elseif(isequal(size(cIn),[1,3]))
    cIn = [cIn(1) 0 cIn(2) 0; 0 cIn(1) cIn(3) 0; 0 0 1 0];
end

cIn = single(cIn);
panoramic = boolean(panoramic(1));
calllib('LibCal','addCamera',cIn, panoramic);

end

