function [ cloud ] = getNorms(cloud, numNeighbours)

%remove non distance related points
cloud = cloud(:,1:3);
cloud(end,4) = 0;

%remove zero elements
cloud = cloud(~((cloud(:,1) == 0) & (cloud(:,2) == 0) & (cloud(:,3) == 0)),:);

%create kdtree
kdtreeobj = KDTreeSearcher(cloud(:,1:3),'distance','euclidean');

%get nearest neighbours
n = knnsearch(kdtreeobj,cloud(:,1:3),'k',(numNeighbours+1));

%remove self
n = n(:,2:end);

for i = 1:size(cloud,1)
    C = zeros(3,3);
    
    for j = 1:numNeighbours
        %get difference
        p = cloud(i,1:3) - cloud(n(i,j),1:3);  
        C = C + (p')*(p);
    end
    
    C = C / numNeighbours;
    
    %get eigen values and vectors
    [v,d] = eig(C);
    d = diag(d);
    
    [~,k] = min(d);
    
    %ensure all points have same direction
    if(v(k,:)*cloud(i,1:3)' < 0)
        norm = v(k,:);
    else
        norm = -v(k,:);
    end
    
    cloud(i,4) = atan2d(norm(1), sqrt(norm(2).^2 + norm(3).^2));
    %cloud(i,5) = atan2d(norm(2), sqrt(norm(1).^2 + norm(3).^2));
    %cloud(i,6) = atan2d(norm(3), sqrt(norm(2).^2 + norm(1).^2));
    
    %store normal values
    %cloud(i,4) = norm(3);
    %cloud(i,5) = norm(2);
    %cloud(i,6) = norm(1);
end

end

