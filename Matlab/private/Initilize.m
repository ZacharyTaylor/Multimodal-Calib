function [] = Initilize(cam,numGen)
%INITILIZE Summary of this function goes here
%   Detailed explanation goes here

numGen = round(numGen(1));
if(numGen < 1)
    error('numGen must be atleast 1');
elseif(numGen > 100)
    warning('large values of numGen use up a lot of GPU memory with very little improvement in performance');
end

if(cam)
    calllib('LibCal','initalizeCamera',numGen);
else
    calllib('LibCal','initalizeImage',numGen);
end

end

