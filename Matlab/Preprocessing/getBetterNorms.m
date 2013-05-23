function [ mag, phase ] = getBetterNorms(data, tform, numInterpolate)

%remove non distance related points
cloud = data(:,1:3);

cloud(:,4) = 0;

%transform points
tform = double(tform);   
tformMat = angle2dcm(tform(6), tform(5), tform(4));
tformMat(4,4) = 1;
tformMat(1,4) = tform(1);
tformMat(2,4) = tform(2);
tformMat(3,4) = tform(3);

cloud = cloud(:,1:4);
cloud(:,4) = 1;

cloud = (tformMat*(cloud'))';

%project points onto sphere
sphere = zeros(size(cloud,1),6);
sphere(:,1) = atan2(cloud(:,1), cloud(:,2));
sphere(:,2) = atan(cloud(:,3)./ sqrt(cloud(:,1).^2 + cloud(:,2).^2));
sphere(:,3) = data(:,4);%sqrt(cloud(:,1).^2 + cloud(:,2).^2 + cloud(:,3).^2);

%get interpolation points
xRange = (max(sphere(:,1)) - min(sphere(:,1)));
yRange = (max(sphere(:,2)) - min(sphere(:,2)));

xSteps = sqrt(numInterpolate * xRange / yRange);
ySteps = numInterpolate / xSteps;
xRange = xRange / xSteps;
yRange = yRange / ySteps;

xRange = min(sphere(:,1)):xRange:max(sphere(:,1));

yRange = min(sphere(:,2)):yRange:max(sphere(:,2));
[qx,qy] = meshgrid(xRange, yRange);

F = TriScatteredInterp(sphere(:,1),sphere(:,2),sphere(:,3));
%qz = griddata(sphere(:,1),sphere(:,2),sphere(:,3),qx,qy);
qz = F(qx,qy);

qz(isnan(qz)) = 0;

% [Nx,Ny,Nz] = surfnorm(qx,qy,qz); 
% 
% img = abs(atan2d(Ny,Nx));
% img(isnan(img)) = 0;
% 
% img = img-min(img(:));
% img = img/max(img(:));
%img = histeq(img);

[mag,phase] = imgrad(qz);
%mag = img;
%phase = img;

%interpolate back to original points
mag = interp2(qx,qy,mag,sphere(:,1),sphere(:,2));
phase = interp2(qx,qy,phase,sphere(:,1),sphere(:,2));

%remove edge points
cut = (sphere(:,1) < (min(sphere(:,1)+0.05))) | ...
    (sphere(:,1) > (max(sphere(:,1)-0.05))) | ...
    (sphere(:,2) < (min(sphere(:,2)+0.005))) | ...
    (sphere(:,2) > (max(sphere(:,2)-0.005)));

mag(cut) = 0;

mag = mag - min(mag);
mag = mag / max(mag);
%mag = histeq(mag);

mag(isnan(mag)) = 0;
phase(isnan(phase)) = 0;

end

