function [] = AddTform( tform )
%ADDTFORM Adds a transform that will be used to project moving scans onto base images
% tformIn array holding transform
% if tformIn is of size (1,6) will assume holding 3D transform of format [x,y,z,rx,ry,rz]
% if tformIn is of size (1,7) will assume holding affine transform [x,y,rotation,scale x,scale y,shear x,shear y] 

tform = single(tform(:,:,1));

if(isequal(size(tform),[1,6]))
    tform = CreateTformMat(tform);
elseif(isequal(size(tform),[1,7]))
    rot = [cosd(tform(3)),-sind(tform(3)),0;sind(tform(3)),cosd(tform(3)),0;0,0,1]; 
    scale = [tform(4),0,0;0,tform(5),0;0,0,1];
    shear = [1,tform(6),0;tform(7),1,0;0,0,1];
    trans = [1,0,tform(1);0,1,tform(2);0,0,1];
    tform = single(trans*shear*scale*rot);
end
        
tformSizeX = size(tform,2);
tformSizeY = size(tform,1);

calllib('LibCal','addTform',tform,tformSizeX,tformSizeY);

end

