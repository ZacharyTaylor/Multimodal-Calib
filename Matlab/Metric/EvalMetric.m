function [ val ] = EvalMetric(imgNum)
%GETMOVE evaluates metric
%imgNum - number of moving scan to compare generated scan against

%check inputs
if((imgNum ~= round(imgNum)) || (imgNum < 0))
    TRACE_ERROR('number of move scans must be a positive integer, returning');
    return;
end

%ensures the library is loaded
CheckLoaded();

%evaluate metric
val = calllib('LibCal','getMetricVal', imgNum);


end

