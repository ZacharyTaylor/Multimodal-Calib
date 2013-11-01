function [ out ] = ScanFeature( data, feature, varargin )
%SCANFEATURE sets scan intensity information to selected feature
% data - input scan format [x,y,z,...]
% feature - feature to use options are intensity, range and normals
% normals feature requires an additional argument, a transform to apply to
% the point cloud first

if(or(strcmp('intensity',feature),strcmp('None',feature)))
    out = single(data);
elseif(or(strcmp('distance',feature),strcmp('range',feature)))
    data(:,4) = 0;
    out = single(data(:,1:4));
    out(:,4) = sqrt(sum(out(:,1:3).^2,2));
elseif(strcmp(feature,'normals'))
    tform = varargin{1};  
    data(:,4) = 0;
    out = GetNorms(data, tform);
end

end

