function [] = SetSingleScanIndex()
%SETSINGLESCANINDEX sets up all the indexes needed for 1 scan 1 image calib

calllib('LibCal','addTformIndex',0,1);
calllib('LibCal','addScanIndex',0,1);
calllib('LibCal','addCameraIndex',0,1);

end

