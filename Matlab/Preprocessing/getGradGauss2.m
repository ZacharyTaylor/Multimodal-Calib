function [ Mag, Phase ] = getGradGauss2( in, tform, sigma )
%GETGRADIENT Summary of this function goes here
%   Detailed explanation goes here
filtSize = 20;

vals = in(:,4);

in = in(:,1:3);
in(:,4) = 1;

%transform points
tform = double(tform);   
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

in = (tformMat*(in'))';

%project points onto sphere
sphere = zeros(size(in,1),2);
sphere(:,1) = atan2(in(:,1), in(:,2));
sphere(:,2) = atan(in(:,3)./ sqrt(in(:,1).^2 + in(:,2).^2));

kdTree = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');
idx = knnsearch(kdTree,kdTree.X,'k',filtSize);

Dx = repmat(kdTree.X(idx(:,1),1),1,filtSize) - reshape(kdTree.X(idx,1),size(idx,1),filtSize);
Dy = repmat(kdTree.X(idx(:,1),2),1,filtSize) - reshape(kdTree.X(idx,2),size(idx,1),filtSize);

f = (1/((sigma^2) * 2 * pi)) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));
fx = (Dx/((sigma^4) * sqrt(2*pi))) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));
fy = (Dy/((sigma^4) * sqrt(2*pi))) .* exp(-(Dx.^2 + Dy.^2)/(2*(sigma.^2)));
    
I = reshape(vals(idx),size(vals,1),filtSize).*f;
I = sum(I,2)./sum(f,2);

Ix = reshape(I(idx),size(vals,1),filtSize).*fx;
Iy = reshape(I(idx),size(vals,1),filtSize).*fy;   

Ix = sum(Ix,2)./sum(f,2);
Iy = sum(Iy,2)./sum(f,2);

Mag=sqrt(Ix .^ 2 + Iy .^ 2);

%give points on edge of scan a value of zero
Dx = sum(Dx,2);
Dy = sum(Dy,2);

dist = abs(Dx) + abs(Dy);
dist = dist - min(dist);
dist = dist / max(dist);
dist = histeq(dist);

Mag(dist > 0.8) = 0;

Magmax=max(Mag(:));
Mag=Mag/Magmax;

Phase = atan2d(Iy,Ix);
end

