function [] = CheckLoaded()
%CHECKLOADED checks if calibration library is loaded and if it isnt loads
%it

if(~libisloaded('Multimodal-Calib'))
    TRACE_INFO('Loading multimodal calibration library');
    
    [notFound, warnings] = loadlibrary('TODO');
    if(warnings ~= '')
        TRACE_WARNINGS(warnings)
    end
end

end

