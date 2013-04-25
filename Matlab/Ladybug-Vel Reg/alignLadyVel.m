function f=alignLadyVel(base, move, pairs, tform, ladybugParam)

    global FIG;
    f = 0;
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);
        
        %get camera name
        cam = mod(i-1,5);
        cam = ['cam' int2str(cam)];
        
        %get transformation matrix
        tformLady = tform + ladybugParam.(cam).offset;
        tformMat = angle2dcm(tformLady(6), tformLady(5), tformLady(4));
        tformMat(4,4) = 1;
        tformMat(1,4) = tformLady(1);
        tformMat(2,4) = tformLady(2);
        tformMat(3,4) = tformLady(3);
        SetTformMatrix(tformMat);
        
        %setup camera
        focal = ladybugParam.(cam).focal;
        centre = ladybugParam.(cam).centre;
        cameraMat = cam2Pro(focal,focal,centre(1),centre(2));
        SetCameraMatrix(cameraMat);
        
        Transform(pairs(i,2)-1);
        InterpolateBaseValues(pairs(i,2)-1);

        temp = EvalMetric(pairs(i,2)-1);
        if(~(isnan(temp) || isinf(temp)))
            f = f + temp;
        end

        %display current estimate
        if(FIG.countMax < FIG.count)
            FIG.count = 0;
            h = gcf;
            sfigure(FIG.fig);

            b = uint8(255*OutputImage(width, height,pairs(i,2)-1));
            se = strel('ball',5,5);
            b = imdilate(b,se);

            comb = uint8(zeros([height width 3]));
            comb(:,:,1) = base{pairs(i,1)}.v;
            comb(:,:,2) = b;

            %subplot(3,5,i); imshow(b);
            %subplot(3,5,5+i); imshow(base{pairs(i,1)}.v);
            %subplot(3,5,10+i); imshow(comb);
            
            subplot(1,3,1); imshow(b);
            subplot(1,3,2); imshow(base{pairs(i,1)}.v);
            subplot(1,3,3); imshow(comb);

            drawnow
            fprintf('current transform:\n     metric = %1.3f\n     rotation = [%1.2f, %1.2f, %1.2f]\n     translation = [%1.2f, %1.2f, %1.2f]\n\n',...
                f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6));

            sfigure(h);
        end
        
        %change so cost function range is 0-1
        f = f / size(pairs,1);
        f = -f;
    end
end