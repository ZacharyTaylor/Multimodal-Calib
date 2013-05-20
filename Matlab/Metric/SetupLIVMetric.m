function [] = SetupLIVMetric(bAvg)
%SETUPLIVMETRIC sets up the mutual information metric

%ensures the library is loaded
CheckLoaded();

imgT = zeros([size(bAvg,2),size(bAvg,1),size(bAvg,3)]);
for i = 1:size(bAvg,3)
    imgT(:,:,i) = bAvg(:,:,i)';
end

calllib('LibCal','setupLIVMetric',imgT, size(imgT,1), size(imgT,2));

end