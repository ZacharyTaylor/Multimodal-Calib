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
    
    if(ispc)
        os = 'dll';
    else
        os = 'so';
    end
       
    Version = ['Multimodal-Calib' arch '.' os];
    
    fprintf('Loading multimodal calibration library\n');
    
    [notFound, warnings] = loadlibrary([pwd '/../Binaries/' Version],[pwd '/../Code/MatCalls.h'],'alias','LibCal');
    
end

end

