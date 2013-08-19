function [] = TRACE_WARNING( string )
%TRACE_INFO Outputs debug warnings

global DEBUG_LEVEL;

if(exist('DEBUG_TRACE','var'))
    if(DEBUG_LEVEL > 1)
        info = dbstack(1);
        info = info(1);
        fprintf('Matlab Warning: %s(%d): ',info.name,info.line)
        fprintf(string);
        fprintf('\n');
    end
end

end

