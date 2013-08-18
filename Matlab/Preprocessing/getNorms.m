function [ data ] = getNorms(data, tform)
%remove non distance related points
cloud = data(:,1:3);

cloud(:,4) = 0;

%transform points
tformMat = CreateTformMat(tform);

cloud = cloud(:,1:4);
cloud(:,4) = 1;

cloud = (tformMat*(cloud'))';

%project points onto sphere
sphere = cloud; zeros(size(cloud,1),6);
%sphere(:,1) = atan2(cloud(:,1), cloud(:,2));
%sphere(:,2) = atan(cloud(:,3)./ sqrt(cloud(:,1).^2 + cloud(:,2).^2));
%sphere(:,3) = sqrt(cloud(:,1).^2 + cloud(:,2).^2 + cloud(:,3).^2);

numNeighbours = 19;

%create kdtree
kdtreeobj = KDTreeSearcher(sphere(:,1:3),'distance','euclidean');

%get nearest neighbours
n = knnsearch(kdtreeobj,sphere(:,1:3),'k',(numNeighbours+1));

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
%     if(v(k,:)*sphere(i,1:3)' < 0)
         norm = v(k,:);
%     else
%         norm = -v(k,:);
%     end
    
    %store normal values
    data(i,4) = abs(atan2(abs(norm(1)),sqrt(norm(2)^2 + norm(3)^2)));
end

data(:,4) = data(:,4) - min(data(:,4));
data(:,4) = data(:,4) / max(data(:,4));
data(:,4) = histeq(data(:,4));

end

