global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(5,5,5);

a = getImages;
a(:,:,2) = a;
a(:,:,3) = a(:,:,1);

%LoadMoveScan(1,a,3);
%out = GetMove(1);

LoadMoveImage(1,a);
LoadBaseImage(2,a);

InterpolateBaseValues(1,2);

out = getGenVals(1);
out2 = GetBase(2);

% SetupCamera(1);
% camera = [100 0 100 0; 0 100 100 0; 0 0 1 0];
% SetCameraMatrix(camera);
% 
% SetupCameraTform();
% tform = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
% SetTformMatrix(tform);
% 
% Transform(1);
% 
% out = getGenerated(1);

ClearLibrary