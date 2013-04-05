global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(5,5,5);

a = getImages;

LoadMoveScan(1,a,3);
out = GetMove(1);

LoadBaseImage(2,a);

SetupCamera(1);
camera = [100 0 100 0; 0 100 100 0; 0 0 1 0];
SetCameraMatrix(camera);

SetupCameraTform();
tform = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
SetTformMatrix(tform);

Transform(1);

out = getGenerated(1);

ClearLibrary