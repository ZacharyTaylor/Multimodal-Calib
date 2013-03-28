function [] = TRACE_INFO( string )
%TRACE_INFO Outputs debug info

global DEBUG_TRACE;

if(exist('DEBUG_TRACE','var'))
    if(DEBUG_TRACE > 0)
        info = dbstack(1);
        info = info(1);
        fprintf('Info: %s(%d): ',info.name,info.line)
        fprintf(string);
        fprintf('\n');
    end
end

end

