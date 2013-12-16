%% User set parameters

%the transform between the camera and the lidar
%can be either a 4x4 transform matrix or [x,y,z,rx,ry,rz] (rotations in
%radians, rotation order rz, ry, rx)
tform = [-0.634307145884011,0.0611205618713079,0.0392814039466259,-1.56704554974731,0.00116227912457255,1.53313471595683];

%camera intrinsic parameters
%can be either 3x4 camera matrix or [f, cx, cy] or [fx, fy, cx, fy]
cam = [770.165286176755,1005,173.413134492138];

%% Setup

%metric to use
metric = 'None';

%get scans and images
%move = GetPointClouds(1);
%base = GetImagesC(1, false);

scan = co;
scan(end,135) = 0;

base = cell(1);

for i = 1:132
    base{1}.v = D(:,:,i);
    base{1}.c = D(:,:,i);
    
    Setup(1,metric, move, base, tform, cam, true);

    %% Colour and output scan
    temp = ColourScan(0);
    scan(:,3+i) = temp(:,4);
    i
end


dlmwrite('ScanOut2.csv',scan,'precision',12 );

%% Clean up
ClearEverything();