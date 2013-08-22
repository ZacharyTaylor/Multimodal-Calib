function [] = CheckLoaded()
%CHECKLOADED checks if calibration library is loaded and if it isnt loads
%it

%library to load depends on your debug options, os and architecture
%named as Multimodal-Calib-<arch><debug>.<os>
%options are
% <arch>: 64 = 64 bit, 32 = 32 bit
% <debug>: O = optimized, outputs errors to matlab
%          W = debug, outputs errors and warnings to matlab
%          I = debug, outputs errors, warnings and info on what its doing 
%              to matlab (very verbose)
% <os>: dll = windows, so = linux

if(~libisloaded('LibCal'))
    
    str = computer;
    if(str(end) == '4')
        arch = '64';
    else
        arch = '32';
    end
    
    if(str(1) == 'P')
        os = 'dll';
    else
        os = 'so';
    end
    
    global DEBUG_LEVEL;
    if(DEBUG_LEVEL == 3)
        debug = 'I';
    elseif(DEBUG_LEVEL == 2)
        debug = 'W';
    else
        debug = 'O';
    end
    
    Version = ['Multimodal-Calib-' arch debug '.' os];
    
    TRACE_INFO('Loading multimodal calibration library');
    
    [notFound, warnings] = loadlibrary([pwd '/../Binaries/' Version],[pwd '/../Code/MatlabCalls.h'],'alias','LibCal');
    
    if(size(notFound,1) ~= 0)
        fprintf(notFound);
    end
    
    if(size(warnings,1) ~= 1)
        fprintf(warnings);
    end
    
    calllib('LibCal','setupCUDADevices');
    
end

end

