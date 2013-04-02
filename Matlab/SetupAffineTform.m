function [] = SetupAffineTform()
%SETUPAFFINETFORM sets up the affine tform

%ensures the library is loaded
CheckLoaded();

calllib('LibCal','SetupAffineTform');

end