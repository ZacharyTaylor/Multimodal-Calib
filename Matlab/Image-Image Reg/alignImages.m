function f=alignImages(base, move, pairs, tform)

    global FIG;
        
    tform = double(tform);
    
    midShift = [1,0,size(move{1}.v,2)/2;0,1,size(move{1}.v,1)/2;0,0,1];
    rot = [cosd(tform(3)),-sind(tform(3)),0;sind(tform(3)),cosd(tform(3)),0;0,0,1]; 
    scale = [tform(4),0,0;0,tform(5),0;0,0,1];
    shear = [1,tform(6),0;tform(7),1,0;0,0,1];
    trans = [1,0,tform(1);0,1,tform(2);0,0,1];
    midBack = [1,0,-size(move{1}.v,2)/2;0,1,-size(move{1}.v,1)/2;0,0,1];

    tformMat = single(midShift*trans*shear*scale*rot*midBack);
    
    SetTformMatrix(tformMat);
    
    f = 0;
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);

        Transform(pairs(i,1)-1);
        InterpolateBaseValues(pairs(i,2)-1);

        temp = EvalMetric(pairs(i,1)-1);

        if(isnan(temp))
            a = 1;
        end
        if(~(isnan(temp) || isinf(temp)))
            f = f + temp;
        end

        %display current estimate
        if(FIG.countMax < FIG.count)
            FIG.count = 0;
            h = gcf;
            sfigure(FIG.fig);

            m = uint8(255*OutputImage(width, height,pairs(i,1)-1,1));

            comb = uint8(zeros([height width 3]));
            comb(:,:,1) = base{pairs(i,1)}.v;
            comb(:,:,2) = m(:,:,1);

            subplot(3,1,1); imshow(m(:,:,1));
            subplot(3,1,2); imshow(base{pairs(i,2)}.v);
            subplot(3,1,3); imshow(comb);

            drawnow
            fprintf('current transform:\n     metric = %1.3f\n     translation = [%3.0f, %3.0f]\n     rotation = %1.2f\n     scale = [%1.2f,%1.2f]\n     shear = [%0.3f, %0.3f]\n\n',...
                f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6),tform(7));

            sfigure(h);
        end
        
        %change so cost function range is 0-1
        f = f / size(pairs,1);
        f = -f;
    end
end