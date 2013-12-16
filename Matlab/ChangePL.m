function [] = ChangePL( base, level, metric, mask )
%CHANGEPL Changes level of optimization pyrimid

%remove current images
ClearImages();

%correct for structs
for i = 1:size(base(:),1)
    if(isstruct(base{i}))
        base{i} = base{i}.v;
    end
end

%filter base images
for i = 1:size(base(:),1)
    base{i} = FilterImage(base{i}, metric, mask{i});
end

%correct for average image in levinson method
if(and(strcmp(metric,'LEV'),size(base(:),1) ~= 1))
    avImg = base{1}(:,:,1);
    for i = 2:size(base(:),1)
        avImg = avImg + base{i}(:,:,1);
    end
    avImg = avImg / size(base(:),1);

    for i = 1:size(base(:),1)
        base{i}(:,:,1) = base{i}(:,:,1) - avImg;
    end
end


%blur images
if(level > 1)
    sd = 2^(level-1);
    G = fspecial('gaussian',[6*sd 6*sd],sd);
    for i = 1:size(base(:),1)
        for z = 1:size(base{i},3)
            base{i}(:,:,z) = imfilter(base{i}(:,:,z),G,'same');
        end
    end
end

%load images
for i = 1:size(base(:),1)
    AddBaseImage(base{i});
end

end

