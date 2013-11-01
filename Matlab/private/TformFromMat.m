function [ tform ] = TformFromMat( tformMat )
%CREATETFORMMAT Takes in tformMat outputs tform

x = tformMat(1,4);
y = tformMat(2,4);
z = tformMat(3,4);

[yaw, pitch, roll] = dcm2angle(tformMat(1:3,1:3)');

tform = [x,y,z,roll,pitch,yaw];
        
end

