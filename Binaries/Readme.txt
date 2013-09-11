The library you want to load depends on your debug options, os and architecture
named as Multimodal-Calib-<arch><debug>.<os>
options are
 <arch>: 64 = 64 bit, 32 = 32 bit
 <debug>: O = optimized, outputs errors to matlab
          W = debug, outputs errors and warnings to matlab
          I = debug, outputs errors, warnings and info on what its doing 
              to matlab (very verbose)
 <os>: dll = windows, a = linux
 
 To compile the 32 bit versions requires a 32 bit install of matlab, currently 
 I only have a 64 bit version thus they are not provided
 Also while it is possible for the code to run on cuda 1.1 systems
 VS wouldn't compile for it so the windows binaries are built with cuda 1.2