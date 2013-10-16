function [] = Initilize(cam)
%INITILIZE Summary of this function goes here
%   Detailed explanation goes here

if(cam)
    calllib('LibCal','initalizeCamera');
else
    calllib('LibCal','initalizeImage');
end

end

