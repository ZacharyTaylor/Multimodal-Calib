function [ result ] = EvalMetric()
%EVALMETRIC Evaluates metric over given images and scans

result = calllib('LibCal','evalMetric');
end

