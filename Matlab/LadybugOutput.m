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
path = 'base path goes here';
%range of images to use
imRange = [1];

offset = [0.0215,0.0026,-0.8822,-0.0014,0.0087,3.1143];

%if saving
saveAll = true;
save = false;

%save path
savePath = 'save path goes here';

%% setup transforms and images
SetupCamera(0);
SetupCameraTform();

[basePaths, movePaths, pairs] = MatchImageScanNav( path, imRange, false );
%read in nav data
[ navPos, navSpeed, navTime ] = ReadNavData( [path 'nav location goes here'] );

Initilize(1, 1);

out = cell(size(pairs,1),1);

for i = 1:5
    temp = imread([path 'mask location goes here' int2str(i-1) '.png']);
    mask(:,:,i) = temp(:,:,1);
end

folder = [path 'cam location goes here'];
files = dir(folder);
fileIndex = find(~[files.isdir]);

time = uint64(zeros(length(fileIndex),1));

%get image times
for i = 1:length(fileIndex)
    %get name
    fileName = files(fileIndex(i)).name;

    %get time from name
    temp = textscan(fileName,'%u64');
    time(i) = temp{1};
end
    
%% get Data
for j = 1:(size(pairs,1))
    
    m = ReadVelData(movePaths{pairs(j,2)});
    LoadMoveScan(0,m,3);
    
    out{j} = zeros(size(m,1),6);
    out{j}(:,1:3) = m(:,1:3);
    
    for k = [1,3,4,5,2]

        b = imread(basePaths{pairs(j,1),k});
        b(repmat(mask(:,:,k) == 0,[1,1,3])) = 0;
        b(b ~=0) = imadjust(b(b ~= 0),stretchlim(b(b~= 0)),[]);
        b = double(b)/255;
               
        LoadBaseImage(0,b);

        %% colour image

        %get camera name
        cam = k-1;
        cam = ['cam' int2str(cam)];

        %baseTform
        tformMatB = CreateTformMat(tform);
        
        %get transformation matrix
        tformLady = ladybugParam.(cam).offset;
        tformMat = CreateTformMat(tformLady);
        tformMat = tformMat/tformMatB;
        SetTformMatrix(tformMat);

        %setup camera
        focal = ladybugParam.(cam).focal;
        centre = ladybugParam.(cam).centre;
        cameraMat = cam2Pro(focal,focal,centre(1),centre(2));
        SetCameraMatrix(cameraMat);

        Transform(0);
        InterpolateBaseValues(0);
        
        %get colour image
        gen = GetGen();
        
        %ignore rear cam
        if(k==2)
            outLoc = out{j};
            outLoc(any(gen(:,4:6),2),4:6) = gen(any(gen(:,4:6),2),4:6);
            %remove black points
            outLoc = outLoc(any(outLoc(:,4:6),2),:);
            
            %remove very bright ponints
            outLoc = outLoc(~any((outLoc(:,4:6) > 150),2),:);
            
            outLoc(4:6) = outLoc(4:6)*255/150;
            
            if(save)
                %save image
                strPly = int2str(imRange(j));
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
    
    %remove very bright ponints
    out{j} = out{j}(~any((out{j}(:,4:6) > 150/255),2),:);
    out{j}(4:6) = out{j}(4:6)*255/150;
    
    %add absolute position
    
    %get time difference of image from nav
    timeDif = time(imRange(j)) - navTime;
    [~, idx] = min(abs(timeDif));
    timeDif = timeDif(idx);

    %get position image was taken at
    tformNav = navPos(idx,:);
    tformNav = tformNav + (double(repmat(timeDif(:),1,6))/1000000) .* navSpeed(idx,:);

    %add offset
    out{j} = transformPoints( offset, out{j}, true);

    %add position
    out{j} = transformPoints( tformNav, out{j}, false);
        
        
    
    fprintf('Processing Image %i\n',j);
end

%combine all points
out = cell2mat(out);

%center at zero, zero
out(:,1) = out(:,1) - mean(out(:,1));
out(:,2) = out(:,2) - mean(out(:,2));
out(:,3) = out(:,3) - mean(out(:,3));

if(saveAll)
    %save image
    strPly = [savePath, 'Scans ' int2str(imRange(1)) ' to ' int2str(imRange(end)), '.ply'];

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