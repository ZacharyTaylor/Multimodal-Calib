function f=alignPoints(base, move, pairs, tform)

    global FIG;
    
    %get transformation matirx    
    tform = double(tform);   
    tformR = pi.*tform(1:3)./180;
    tformMat = angle2dcm(tformR(3), tformR(2), tformR(1));
    tformMat(4,4) = 1;
    tformMat(1,4) = tform(4);
    tformMat(2,4) = tform(5);
    tformMat(3,4) = tform(6);
    
    SetTformMatrix(tformMat);
    
    f = 0;
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);

        Transform(pairs(i,1)-1);
        InterpolateBaseValues(pairs(i,2)-1);

        temp = EvalMetric(pairs(i,1)-1);
        if(~(isnan(temp) || isinf(temp)))
            f = f + temp;
        end

        %display current estimate
        if(FIG.countMax < FIG.count)
            FIG.count = 0;
            h = gcf;
            sfigure(FIG.fig);

            b = uint8(255*OutputImage(width, height,pairs(i,1)-1));

            comb = uint8(zeros([height width 3]));
            comb(:,:,1) = base{pairs(i,1)}.v;
            comb(:,:,2) = b;

            subplot(3,1,1); imshow(b);
            subplot(3,1,2); imshow(base{pairs(i,2)}.v);
            subplot(3,1,3); imshow(comb);

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