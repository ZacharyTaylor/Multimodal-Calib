function [ loc, vals ] = ImageToList( img )
%IMAGETOLIST Summary of this function goes here
%   Detailed explanation goes here
img = double(img);

x = repmat((1:size(img,2)),size(img,1),1);
y = repmat((1:size(img,1))',1,size(img,2));

loc = [x(:), y(:)];
vals = img(:);

end

