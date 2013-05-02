function [ mag, phase ] = getBetterNorms(cloud, numNeighbours, numInterpolate)

%remove non distance related points
cloud = cloud(:,1:3);

cloud(:,6) = 0;

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

[Nx,Ny,Nz] = surfnorm(qx,qy,qz); 

%interpolate back to original points
sphere(:,4) = interp2(qx,qy,Nx,sphere(:,1),sphere(:,2));
sphere(:,5) = interp2(qx,qy,Ny,sphere(:,1),sphere(:,2));
sphere(:,6) = interp2(qx,qy,Nz,sphere(:,1),sphere(:,2));

%get gradient and magnitude
phase = atan2d(sphere(:,5),sphere(:,4));
mag = ones(size(sphere(:,6),1),1);%-sphere(:,6);

end

