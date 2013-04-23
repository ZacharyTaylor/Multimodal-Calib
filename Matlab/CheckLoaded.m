function [] = CheckLoaded()
%CHECKLOADED checks if calibration library is loaded and if it isnt loads
%it

if(~libisloaded('LibCal'))
    TRACE_INFO('Loading multimodal calibration library');
    
    [notFound, warnings] = loadlibrary('../Code/Multimodal-Calib/x64/Fast/Multimodal-Calib.dll','../Code/MatlabCalls.h','alias','LibCal');
end

end

