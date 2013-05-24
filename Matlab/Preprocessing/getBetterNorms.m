function [ mag, phase ] = getBetterNorms(cloud, tform, numInterpolate)

%remove non distance related points
cloud = cloud(:,1:3);

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
sphere(:,3) = sqrt(cloud(:,1).^2 + cloud(:,2).^2 + cloud(:,3).^2);

%get interpolation points
xRange = (max(sphere(:,1)) - min(sphere(:,1)));
yRange = (max(sphere(:,2)) - min(sphere(:,2)));
step = sqrt(numInterpolate / (xRange*yRange));
xRange = 1 / (xRange * step);
yRange = 1 / (yRange * step);

xRange = min(sphere(:,1)):xRange:max(sphere(:,1));

yRange = min(sphere(:,2)):yRange:max(sphere(:,2));
[qx,qy] = meshgrid(xRange, yRange);

F = TriScatteredInterp(sphere(:,1),sphere(:,2),sphere(:,3));
%vq = griddata(sphere(:,1),sphere(:,2),sphere(:,3),xq,yq);
qz = F(qx,qy);

%[Nx,Ny,Nz] = surfnorm(qx,qy,qz); 

%img = atan2d(Ny,Nx);
%img(isnan(img)) = 0;

qz(isnan(qz)) = 0;

[mag,phase] = imgrad(qz);


%interpolate back to original points
mag = interp2(qx,qy,mag,sphere(:,1),sphere(:,2));
phase = interp2(qx,qy,phase,sphere(:,1),sphere(:,2));

%increase weight of close points
mag = mag - min(mag);
mag = mag / max(mag);
%mag = histeq(mag);

mag(isnan(mag)) = 0;
phase(isnan(phase)) = 0;

end

