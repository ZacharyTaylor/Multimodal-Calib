function [] = SetupCameraTform()
%SETUPAFFINETFORM sets up the camera tform

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','setupCameraTform');

end