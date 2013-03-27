function [] = TRACE_WARNING( string )
%TRACE_INFO Outputs debug warnings

global DEBUG_TRACE;

if(exist('DEBUG_TRACE','var'))
    if(DEBUG_TRACE > 1)
        info = dbstack(1);
        fprintf('WARNING: %s(%d): ',info.name,info.line)
        fprintf(string);
        fprintf('\n');
    end
end

end

