function [] = SetupIndex(moveSize, baseSize, tformSize, cameraSize )
%SETUPINDEX Sets up the assosiation between scans, images, transforms and
%cameras
% This setup assumes there is atleast as many images as scans, other
% combinations are possible but will need to be setup manually
%
% let base be a cell holding n by m images.
% it is assumed each row of images are images taken at the same time and
% thus all belonging to one scan.
% thus move is a n by 1 cell holding the scans with each scan being related
% to a row of images
% if only one transform exists it is applied to all the scans
% if the transform is a 1 by m cell then this transform is applied to
% each scan in the same column
% if the transform is a n by 1 cell then this transform is applied to each
% scan in the same row
% if the transform is a n by m cell then 1 transform maps to 1 image
% the same rules for transforms apply to cameras

%set scan index
if(moveSize(1) == 1)
    idx = zeros(baseSize(1),baseSize(2));
else
    idx = repmat((0:(baseSize(1)-1))',1,baseSize(2));
end
calllib('LibCal','addScanIndex',idx(:),size(idx(:),1));

%set tform index
if(tformSize(1) == 1)
    if(tformSize(2) == 1)
        idx = zeros(baseSize(1),baseSize(2));
    else
        idx = repmat(0:(baseSize(2)-1),baseSize(1),1);
    end
else
    if(tformSize(2) == 1)
        idx = repmat((0:(baseSize(1)-1))',1,baseSize(2));
    else
        idx = sub2ind(baseSize, 1:baseSize(2), 1:baseSize(1))-1;
    end
end
calllib('LibCal','addTformIndex',idx(:),size(idx(:),1));

if(cameraSize ~= 0)
    %set camera index
    if(cameraSize(1) == 1)
        if(cameraSize(2) == 1)
            idx = zeros(baseSize(1),baseSize(2));
        else
            idx = repmat(0:(baseSize(2)-1),baseSize(1),1);
        end
    else
        if(cameraSize(2) == 1)
            idx = repmat((0:(baseSize(1)-1))',1,baseSize(2));
        else
            idx = sub2ind(baseSize, 1:baseSize(2), 1:baseSize(1))-1;
        end
    end
    calllib('LibCal','addCameraIndex',idx(:),size(idx(:),1));
end

end

