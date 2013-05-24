loadPaths;
global DEBUG_TRACE
DEBUG_TRACE = 3;

Initilize(5,5);
SetupMIMetric();

a = getImages;

LoadMoveImage(1,a);
LoadBaseImage(2,a);

% 
% out = EvalMetric(1);

%out = getGenVals(1);
%out2 = GetBase(2);

% SetupCamera(1);
% camera = [100 0 100 0; 0 100 100 0; 0 0 1 0];
% SetCameraMatrix(camera);

SetupAffineTform();
tform = [1 1 15 ; 0 1 0 ; 0 0 1];
SetTformMatrix(tform);

Transform(1);

InterpolateBaseValues(2);

genM = GetMove(1);
genP = GetGen(1);

out2 = OutputImage(1500,400);
imshow(out2,[0 255]);
% 
% out = getGenerated(1);

%% cleanup
ClearLibrary;
rmPaths;