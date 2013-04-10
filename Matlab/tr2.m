global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(1,1,1);

a = getImages;

LoadMoveImage(1,a);
LoadBaseImage(1,a);

SetupAffineTform();

tform = [2 0 0; 0 2 0; 0 0 1];
SetTformMatrix(tform);

Transform(1);

out = getGenerated(1);

ClearLibrary