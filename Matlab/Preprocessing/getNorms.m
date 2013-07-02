function [ data ] = getNorms(data, tform)
%remove non distance related points
cloud = data(:,1:3);

cloud(:,4) = 0;

%transform points
tform = double(tform);   
tformMat = createTformMat(tform);

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
    
    %get difference
    p = repmat(sphere(i,1:3),numNeighbours,1) - sphere(n(i,:),1:3);  
    C = (p')*(p);
        
    C = C / numNeighbours;
    
    %get eigen values and vectors
    [v,d] = eig(C);
    d = diag(d);
    
    [~,k] = min(d);
    
    %ensure all points have same direction
    %if(v(k,:)*sphere(i,1:3)' < 0)
        norm = v(k,:);
%     else
%         norm = -v(k,:);
%     end
    
    %store normal values
    data(i,4) = atan2(abs(norm(1)),abs(norm(2)));
end

data(:,4) = data(:,4)-min(data(:,4));
data(:,4) = data(:,4)/max(data(:,4));

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
% 
% end

