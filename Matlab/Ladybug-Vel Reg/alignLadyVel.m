function f=alignLadyVel(base, move, pairs, tform, ladybugParam)

    global FIG;
    f = zeros(size(tform,1),1);
    
    setTformNum(size(tform,1));
    SetupCameraTform();
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);
        
        %get camera name
        cam = mod(i-1,5);
        cam = ['cam' int2str(cam)];
                
        for j = 1:size(tform,1)
            %baseTform
            tformMatB = CreateTformMat(tform(j,:));

            %get transformation matrix
            tformLady = ladybugParam.(cam).offset;
            tformMat = CreateTformMat(tformLady);
            tformMat = tformMat/tformMatB;
            SetTformMatrix(tformMat, j-1);
        end
        
        %setup camera
        focal = ladybugParam.(cam).focal;
        centre = ladybugParam.(cam).centre;
        cameraMat = cam2Pro(focal,focal,centre(1),centre(2));
        SetCameraMatrix(cameraMat);
        
        Transform(pairs(i,2)-1);
        InterpolateBaseValues(pairs(i,1)-1);

        temp = EvalMetric(pairs(i,2)-1, size(tform,1));
        temp(isnan(temp)) = 0;
        temp(isinf(temp)) = 0;
        f = f + temp;

        %display current estimate
        if(FIG.countMax < FIG.count)
            FIG.count = 0;
            h = gcf;
            sfigure(FIG.fig);
            
            b = OutputImage(width, height,pairs(i,2)-1,4);
            b = b(:,:,1);
            b(b ~=0) = histeq(b(b~=0));
            b = uint8(255*b);          


            comb = uint8(zeros([height width 3]));
            comb(:,:,1) = base{pairs(i,1)}.v;
            comb(:,:,2) = b;
            
            subplot(1,3,1); imshow(b);
            subplot(1,3,2); imshow(base{pairs(i,1)}.v);
            subplot(1,3,3); imshow(comb);

            drawnow
            fprintf('current transform:\n     metric = %1.3f\n     translation = [%1.2f, %1.2f, %1.2f]\n     rotation = [%1.2f, %1.2f, %1.2f]\n\n',...
                (-f/i),tform(1),tform(2),tform(3),(180*tform(4)/pi),(180*tform(5)/pi),(180*tform(6)/pi));

            sfigure(h);
        end
    end
    
    %change so cost function range is 0-1
    f = f / size(pairs,1);
    f = -f;
end