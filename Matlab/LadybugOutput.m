%% Setup
loadPaths;
set(0,'DefaultFigureWindowStyle','docked');
clc;

global DEBUG_TRACE
DEBUG_TRACE = 2;

%get ladybug parameters
ladybugParam = LadybugConfig;

%% input values

%parameters (x, y ,z, rX, rY, rZ) (rotate then translate,
%rotation order ZYX)
tform = ladybugParam.offset;

%base path
path = 'C:\Data\Almond\';
%range of images to use
imRange = [2, 100, 200, 300, 400 500];

%if saving
save = true;

%save path
savePath = 'C:\Data\Almond\Out\';

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

[basePaths, movePaths, pairs] = MatchImageScan( path, imRange, false );
[ pos  ] = GetNavVals( path, basePaths );

Initilize(1, 1);

out = cell(size(pairs,1),1);

%% get Data
for j = 1:(size(pairs,1))
    
    m = dlmread(movePaths{pairs(j,2)},',');
    LoadMoveScan(0,m,3);
    
    out{j} = zeros(size(m,1),6);
    out{j}(:,1:3) = m(:,1:3);
    
    for k = [1,3,4,5,2]

        b = imread(basePaths{pairs(j,1),k});
        for q = 1:size(b,3)
            temp = b(:,:,q);
            temp(temp ~= 0) = histeq(temp(temp~=0));
            b(:,:,q) = temp;
        end
        b = double(b)/255;
        LoadBaseImage(0,b);

        %% colour image

        %get camera name
        cam = k-1;
        cam = ['cam' int2str(cam)];

        %baseTform
        tformMatB = angle2dcm(tform(6), tform(5), tform(4));
        tformMatB(4,4) = 1;
        tformMatB(1,4) = tform(1);
        tformMatB(2,4) = tform(2);
        tformMatB(3,4) = tform(3);
        
        %get transformation matrix
        tformLady = ladybugParam.(cam).offset;
        tformMat = angle2dcm(tformLady(6), tformLady(5), tformLady(4));
        tformMat(4,4) = 1;
        tformMat(1,4) = tformLady(1);
        tformMat(2,4) = tformLady(2);
        tformMat(3,4) = tformLady(3);
        
        tformMat = tformMat*tformMatB;
        
        SetTformMatrix(tformMat);

        %setup camera
        focal = ladybugParam.(cam).focal;
        centre = ladybugParam.(cam).centre;
        cameraMat = cam2Pro(focal,focal,centre(1),centre(2));
        SetCameraMatrix(cameraMat);

        Transform(0);
        InterpolateBaseValues(0);
        
        %output for debugging
        %b = OutputImage(1232, 1616,0,2);
        %figure,imshow(b);
        
        %get colour image
        gen = GetGen();
        
        %ignore rear cam
        if(k==2)
            outLoc = out{j};
            outLoc(any(gen(:,4:6),2),4:6) = gen(any(gen(:,4:6),2),4:6);
            %remove black points
            outLoc = outLoc(any(outLoc(:,4:6),2),:);
            
            if(save)
                %save image
                strPly = int2str(j);
                strPly = [savePath, strPly, '.ply'];

                if (size(outLoc,1) > 1000)
                    clear output;
                    output.vertex.x = outLoc(:,1);
                    output.vertex.y = outLoc(:,2);
                    output.vertex.z = outLoc(:,3);
                    output.vertex.red = outLoc(:,4);
                    output.vertex.green = outLoc(:,5);
                    output.vertex.blue = outLoc(:,6);
                    ply_write(output,strPly,'binary_little_endian');
                end
            end
        else
            %add colours to output
            out{j}(any(gen(:,4:6),2),4:6) = gen(any(gen(:,4:6),2),4:6);
        end
  
    end
    
    %remove black points
    out{j} = out{j}(any(out{j}(:,4:6),2),:);
    
    %add absolute position
    %get transformation matrix
    tformMat = angle2dcm(pos(j,4), pos(j,5), pos(j,6),'XYZ');
    tformMat(4,4) = 1;
    tformMat(1,4) = pos(j,1);
    tformMat(2,4) = pos(j,2);
    tformMat(3,4) = pos(j,3);
    temp = ones(size(out{j},1),4);
    temp(:,1:3) = out{j}(:,1:3);
    temp = (tformMat*tformMatB*temp')';
    out{j}(:,1:3) = temp(:,1:3);
    
    fprintf('Processing Image %i\n',j);
end

%combine all points
out = cell2mat(out);

%center at zero, zero
out(:,1) = out(:,1) - mean(out(:,1));
out(:,2) = out(:,2) - mean(out(:,2));
out(:,3) = out(:,3) - mean(out(:,3));

if(save)
    %save image
    strPly = int2str(j);
    strPly = [savePath, 'all', '.ply'];

    if (size(out,1) > 1000)
        clear output;
        output.vertex.x = out(:,1);
        output.vertex.y = out(:,2);
        output.vertex.z = out(:,3);
        output.vertex.red = out(:,4);
        output.vertex.green = out(:,5);
        output.vertex.blue = out(:,6);
        ply_write(output,strPly,'binary_little_endian');
    end
end

        
%% cleanup
ClearLibrary;
rmPaths;