function [] = TRACE_ERROR( string )
%TRACE_INFO Outputs debug errors

global DEBUG_LEVEL;

if(exist('DEBUG_TRACE','var'))
    if(DEBUG_LEVEL > 2)
        info = dbstack(1);
        info = info(1);
        fprintf('Matlab Error: %s(%d): ',info.name,info.line)
        fprintf(string);
        fprintf('\n');
    end
end

end

