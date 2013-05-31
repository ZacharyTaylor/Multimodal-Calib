function f=alignPoints(base, move, pairs, tform)

    global FIG;
    
    cameraMat = cam2Pro(tform(7),tform(7),size(base{1}.v,2)/2,size(base{1}.v,1)/2);
    SetCameraMatrix(cameraMat);

    %get transformation matirx    
    tform = double(tform);   
    tformMat = angle2dcm(tform(6), tform(5), tform(4));
    tformMat(4,4) = 1;
    tformMat(1,4) = tform(1);
    tformMat(2,4) = tform(2);
    tformMat(3,4) = tform(3);
    
    SetTformMatrix(tformMat);
    
    f = 0;
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);

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

            b = OutputImage(width, height,pairs(i,2)-1,2);
            b = b(:,:,1);
            %b(b ~=0) = histeq(b(b~=0));
            b = uint8(255*b);

            comb = uint8(zeros([height width 3]));
            comb(:,:,1) = base{pairs(i,1)}.v;
            comb(:,:,2) = b;

            subplot(3,1,1); imshow(b);
            subplot(3,1,2); imshow(base{pairs(i,1)}.v);
            subplot(3,1,3); imshow(comb);

            drawnow
            fprintf('current transform:\n     metric = %1.3f\n     translation = [%1.2f, %1.2f, %1.2f]\n     rotation = [%1.2f, %1.2f, %1.2f]\n\n',...
                f,tform(1),tform(2),tform(3),(180*tform(4)/pi),(180*tform(5)/pi),(180*tform(6)/pi));

            sfigure(h);
        end
        
        %change so cost function range is 0-1
        f = f / size(pairs,1);
        f = -f;
    end
end