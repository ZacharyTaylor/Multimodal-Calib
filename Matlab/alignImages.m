function f=alignImages(base, move, pairs, tform)

    global FIG;
        
    tform = double(tform);
    
    rot = [cosd(tform(3)),-sind(tform(3)),0;sind(tform(3)),cosd(tform(3)),0;0,0,1]; 
    scale = [tform(4),0,0;0,tform(5),0;0,0,1];
    shear = [1,tform(6),0;tform(7),1,0;0,0,1];
    trans = [1,0,tform(1);0,1,tform(2);0,0,1];

    tformMat = single(trans*shear*scale*rot);
    
    SetTformMatrix(tformMat);
    
    f = 0;
    
    for i = 1:size(pairs,1)
        FIG.count = FIG.count + 1;

        width = size(move{pairs(i,1)}.v,2);
        height = size(move{pairs(i,1)}.v,1);

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

            b = OutputImage(width, height);

            comb = zeros([height width 3]);
            comb(:,:,1) = move{pairs(i,1)}.v;
            comb(:,:,2) = b;

            subplot(3,1,1); imshow(b);
            subplot(3,1,2); imshow(move{pairs(i,2)}.v);
            subplot(3,1,3); imshow(comb);

            drawnow
            fprintf('current transform:\n     metric = %1.3f\n     translation = [%3.0f, %3.0f]\n     rotation = %1.2f\n     scale = [%1.2f,%1.2f]\n     shear = [%0.3f, %0.3f]\n\n',...
                f,tform(1),tform(2),tform(3),tform(4),tform(5),tform(6),tform(7));

            sfigure(h);
        end
        
        %change so cost function range is 0-1
        f = f / size(pairs,1);
        f = 2 - f;
    end
end