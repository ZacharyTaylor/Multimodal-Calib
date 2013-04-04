global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(5,5,5);

a = getImages;

LoadMoveImage(1,a);
out = GetMove(1);

LoadBaseImage(2,a);

SetupAffineTform();

% 
tform = [1 0 0; 0 1 0; 0 0 1];
SetTformMatrix(tform);

Transform(1);

out = getGenerated(1);

ClearLibrary