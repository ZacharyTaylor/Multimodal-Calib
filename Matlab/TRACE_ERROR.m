function [] = TRACE_ERROR( string )
%TRACE_INFO Outputs debug errors

global DEBUG_TRACE;

if(exist('DEBUG_TRACE','var'))
    if(DEBUG_TRACE > 2)
        info = dbstack(1);
        info = info(1);
        fprintf('Error: %s(%d): ',info.name,info.line)
        fprintf(string);
        fprintf('\n');
    end
end

end

