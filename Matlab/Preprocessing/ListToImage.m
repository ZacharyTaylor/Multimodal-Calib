function [ img ] = ListToImage( locs, vals , maxX, maxY)
%LISTTOIMAGE Summary of this function goes here
%   Detailed explanation goes here
%locs = [locs(:,2), locs(:,1)];
locs = round(locs);

pos = max(locs);
x = min(pos(1),maxX);
y = min(pos(2),maxY);

img = zeros(y,x);

bounds = (locs(:,2) >= 1) & (locs(:,2) <= maxY) & (locs(:,1) >= 1) & (locs(:,1) <= maxX);
locs = locs(bounds,:);
vals = vals(bounds,:);

img(locs(:,2) + y*(locs(:,1)-1)) = vals;
end

