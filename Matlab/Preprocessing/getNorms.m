function [ data ] = getNorms(data, tform, numInterpolate)
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
sphere(:,3) = sqrt(cloud(:,1).^2 + cloud(:,2).^2 + cloud(:,3).^2);

numNeighbours = 9;

%create kdtree
kdtreeobj = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');

%get nearest neighbours
n = knnsearch(kdtreeobj,sphere(:,1:2),'k',(numNeighbours+1));

%remove self
n = n(:,2:end);

for i = 1:size(sphere,1)
    C = zeros(3,3);
    
    for j = 1:numNeighbours
        %get difference
        p = sphere(i,1:3) - sphere(n(i,j),1:3);  
        C = C + (p')*(p);
    end
    
    C = C / numNeighbours;
    
    %get eigen values and vectors
    [v,d] = eig(C);
    d = diag(d);
    
    [~,k] = min(d);
    
    %ensure all points have same direction
    if(v(k,:)*sphere(i,1:3)' < 0)
        norm = v(k,:);
    else
        norm = -v(k,:);
    end
    
    %store normal values
    data(i,4) = abs(norm(1)+norm(2));%abs(atan2(norm(1),norm(2)));
end

% %get interpolation points
% xRange = (max(sphere(:,1)) - min(sphere(:,1)));
% yRange = (max(sphere(:,2)) - min(sphere(:,2)));
% 
% xSteps = sqrt(numInterpolate * xRange / yRange);
% ySteps = numInterpolate / xSteps;
% xRange = xRange / xSteps;
% yRange = yRange / ySteps;
% 
% xRange = min(sphere(:,1)):xRange:max(sphere(:,1));
% 
% yRange = min(sphere(:,2)):yRange:max(sphere(:,2));
% [qx,qy] = meshgrid(xRange, yRange);
% 
% F = TriScatteredInterp(sphere(:,1),sphere(:,2),sphere(:,3));
% %qz = griddata(sphere(:,1),sphere(:,2),sphere(:,3),qx,qy);
% qz = F(qx,qy);
% 
% qz(isnan(qz)) = 0;
% 
% [Nx,Ny,Nz] = surfnorm(qx,qy,qz); 
% 
% img = abs(atan2d(Ny,Nx));
% img(isnan(img)) = 0;
% 
% img = img-min(img(:));
% img = img/max(img(:));
% 
% %interpolate back to original points
% data(:,4) = interp2(qx,qy,img,sphere(:,1),sphere(:,2));

end

