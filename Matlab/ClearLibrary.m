function [] = ClearLibrary()
%CHECKLOADED clears memory and unloads library
%it

if(libisloaded('LibCal'))
    TRACE_INFO('Clearing multimodal calibration library');
    
    unloadlibrary('LibCal');
end

end

