function [ points ] = livLidar(cloud)
%gets distance to each point

y = 0.5;

cloud(:,4) = sqrt(cloud(:,1).^2+cloud(:,2).^2 + cloud(:,3).^2);

%sort points
sphere(:,1) = atan2(cloud(:,1), cloud(:,2));
sphere(:,2) = 10*atan(cloud(:,3)./ sqrt(cloud(:,1).^2 + cloud(:,2).^2));

kdTree = KDTreeSearcher(sphere(:,1:2),'distance','euclidean');
idx = knnsearch(kdTree,kdTree.X,'k',3);

%remove self
idx = idx(:,2:end);

points = cloud;
points(:,4) = max(abs(cloud(:,4)-cloud(idx(:,1),4)), abs(cloud(:,4)-cloud(idx(:,2),4)));

%points = cloud(2:end-1,:);
%points(:,4) = max(abs(cloud(1:end-2,4) - cloud(2:end-1,4)), abs(cloud(3:end,4) - cloud(2:end-1,4)));

points(:,4) = points(:,4).^y;

points = points(points(:,4) > 0.3,:);

end

