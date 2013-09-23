function [] = setTformNum( numTform )
%SETTFORMNUM Sets up memory for required number of transforms
%   numTform = number of transforms

%load the library
CheckLoaded();

%check inputs
if((numTform ~= round(numTform)) || (numTform < 0))
    TRACE_ERROR('number of transforms must be a positive integer');
    numTform = 1;
end

calllib('LibCal','setNumTforms', numTform);

end

