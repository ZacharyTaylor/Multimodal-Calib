% Program to experimentally test how close an inital guess is required to
% perform local optimization

% This test uses the Ford Campus Vison and Lidar dataset 
% http://robots.engin.umich.edu/SoftwareData/Ford

%% User set parameters

%metric to use, can be one of the following
% SSD - normalized SSD metric
% MI - mutual information (uses a histogram method with 50 bins)
% NMI - normalized mutual information (uses same method as above)
% GOM - gradient orientation measure
% GOMS - modified GOM method that evalutes statistical significants of
% results. Usually outperforms GOM
% LEV - levinson's method
% None - no metric assigned used in generating images and coloured scans
% Note (NMI, MI, LEV and GOM) multiplied by -1 to give minimums in
% optimization
metrics{1,1} = 'GOM';
metrics{4,1} = 'LEV';
metrics{2,1} = 'MI';
metrics{3,1} = 'NMI';
metrics{5,1} = 'GOMS';

%range around transform where the true solution lies 
%can be the following forms
%[x,y,z]
%[x,y,z,rx,ry,rz]
%[x,y,z,rx,ry,rz,f,cx,cy] if camera of form [f,cx,cy]
%[x,y,z,rx,ry,rz,fx,fy,cx,cy] if camera of form [fx,fy,cx,cy]
%rotations in radians, rotation order rx, ry, rz)
range = [0.5 0.5 0.5 0.1 0.1 0.1];

%Sets the update rate in seconds of the output that can be used to evaluate
%the metrics progress. Updating involves transfering the whole image off
%the gpu and so for large scans causes a significant slow down (increase
%value to reduce this issue)
updatePeriod = inf;

%How much to dialate each point by when generating an image from it (only
%effects view generated in updates, does not change metric values)
dilate = 3;

%Number of scans to use in calibration
maxScans = 10;

%feature to use as intensity information of lidar scans. Options are 
%intensity - basic lidar intensity
%range - distance of points from the lidar
%normals - angle between line from lidar to point and a horizontal plane
feature = 'intensity';

%True for panoramic camera, false otherwise
panFlag = false;

%Path to Ford dataset
fordPath = '/home/z/Documents/Datasets/mi-extrinsic-calib-data/';

%Number of Cameras used
numCams = 5;

%% Setup

%get config file
config = FordConfig;

%% Start tests

outTable = cell(size(metrics,1),1);
for m = 1:size(metrics,1);
    metric = metrics{m};
    %from 1 to 20 scans
    scansTable = cell(maxScans,1);
    for numScans = 1:maxScans   
        %number of moving scans
        numMove = numScans;
        %number of base images
        numBase = numScans;

        %change coordinates
        coChangeTable = cell(6,1);
        for coChange = 1:6
            %sweep coordinate differences
            sweepTable = cell(11,1);
            for sweep = 1:11
                if(coChange < 4)
                    dif = 0.1*(sweep - 6);
                else
                    dif = 0.05*(sweep - 6);
                end

                initalGuess = config.T;
                initalGuess(coChange) = initalGuess(coChange) + dif;

                %perform each test 10 times
                res = zeros(10,6);
                for times = 1:10

                    %get scans and images
                    folder = [fordPath config.cam{1}.Path];
                    files = dir(folder);
                    fileIndex = find(~[files.isdir]);

                    scanIdx = randperm(length(fileIndex));
                    scanIdx = sort(scanIdx(1:numScans));

                    base = cell(length(scanIdx),numCams);

                    for j = 1:numCams
                        folder = [fordPath config.cam{j}.Path];
                        files = dir(folder);
                        fileIndex = find(~[files.isdir]);

                        mask = imread([fordPath config.cam{j}.Mask]);
                        for i = 1:length(scanIdx)
                            fileName = files(fileIndex(scanIdx(i))).name;
                            base{i,j} = imread([folder fileName]);

                            %masks edges interfer with levinson method better results without
                            %them
                            if(~strcmp(metric,'LEV'))
                                base{i,j}(mask == 0) = 0;
                            end
                        end
                    end

                    move = ReadFordVelData( fordPath, scanIdx);

                    tform = cell(1,5);
                    multiCamTform = cell(1,5);
                    for i = 1:numCams
                        multiCamTform{1,i} = config.cam{i}.T';
                        tform{1,i} = multiCamTform{1,i}\CreateTformMat(initalGuess);
                    end

                    %get features for scans
                    for i = 1:size(move,1)
                        move{i} = ScanFeature(move{i}, feature, tform);
                    end

                    %get camera
                    cam = cell(1,numCams);
                    for i = 1:numCams
                        cam{i} = config.cam{i}.K;
                    end

                    Setup(10,metric, move, base, tform, cam, panFlag);

                    %ChangePL( base, 5, metric );

                    %% Evaluate metric and Optimize
                    res(times,:) = ConvexOptimize( initalGuess, updatePeriod, dilate, multiCamTform );
                end
                sweepTable{sweep} = res;
                sweep
            end
            coChangeTable{coChange} = sweepTable;
            coChange
        end
        scansTable{numScans} = coChangeTable;
        numScans
    end
    outTable{m} = scansTable;
    save('out','outTable');
    m
end
    

%% Clean up
ClearEverything();