function [ val ] = EvalMetric(imgNum, numResults)
%GETMOVE evaluates metric
%imgNum - number of moving scan to compare generated scan against

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning');
    return;
end
if((numResults ~= round(numResults)) || (numResults < 1))
    TRACE_ERROR('number of results must be a positive integer, returning');
    return;
end

%ensures the library is loaded
CheckLoaded();

val = single(zeros(numResults,1));
valp = libpointer('singlePtr',val);

%evaluate metric
calllib('LibCal','getMetricVal', imgNum,valp);
val = valp.Value;

end

