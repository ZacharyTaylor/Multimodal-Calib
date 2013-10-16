function [] = SetupLib()
%SETUPLIB checks if calibration library is loaded and if it isnt loads
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
    
    fprintf('Loading multimodal calibration library\n');
    
    %[notFound, warnings] = loadlibrary([pwd '/../Binaries/' Version],[pwd '/../Code/MatlabCalls.h'],'alias','LibCal');
    [notFound, warnings] = loadlibrary('C:\Users\ztay8963\Documents\GitHub\Multimodal-Calib\VS\Multimodal-Calib 2.0\x64\Debug\Multimodal-Calib.dll',...
        'C:\Users\ztay8963\Documents\GitHub\Multimodal-Calib\Code\New\MatCalls.h',...
        'alias','LibCal');
    
end

end

