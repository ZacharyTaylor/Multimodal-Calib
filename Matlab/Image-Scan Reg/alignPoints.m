function f=alignPoints(base, move, pairs, tform)

    global FIG;
    tform = double(tform');
    f = zeros(size(tform,2),1);
    
    for k = 1:size(tform,2)
    
        cameraMat = cam2Pro(tform(7,k),tform(7,k),size(base{1}.v,2)/2,size(base{1}.v,1)/2);
        SetCameraMatrix(cameraMat);

        %get transformation matirx       
        tformMat = angle2dcm(tform(6,k), tform(5,k), tform(4,k));
        tformMat(4,4) = 1;
        tformMat(1,4) = tform(1,k);
        tformMat(2,4) = tform(2,k);
        tformMat(3,4) = tform(3,k);

        SetTformMatrix(tformMat);

        for i = 1:size(pairs,1)

            width = size(base{pairs(i,1)}.v,2);
            height = size(base{pairs(i,1)}.v,1);

            Transform(pairs(i,2)-1);
            InterpolateBaseValues(pairs(i,2)-1);

            temp = EvalMetric(pairs(i,2)-1);
            if(~(isnan(temp) || isinf(temp)))
                f(k) = f(k) + temp;
            end

            %change so cost function range is 0-1
            f(k) = f(k) / size(pairs,1);
            f(k) = -f(k);
        end
    end
    
    [~,k] = min(f);
    
    cameraMat = cam2Pro(tform(7,k),tform(7,k),size(base{1}.v,2)/2,size(base{1}.v,1)/2);
    SetCameraMatrix(cameraMat);

    %get transformation matirx       
    tformMat = angle2dcm(tform(6,k), tform(5,k), tform(4,k));
    tformMat(4,4) = 1;
    tformMat(1,4) = tform(1,k);
    tformMat(2,4) = tform(2,k);
    tformMat(3,4) = tform(3,k);

    SetTformMatrix(tformMat);

    for i = 1:size(pairs,1)

        width = size(base{pairs(i,1)}.v,2);
        height = size(base{pairs(i,1)}.v,1);

        Transform(pairs(i,2)-1);
        InterpolateBaseValues(pairs(i,2)-1);
    end
    
    h = gcf;
    sfigure(FIG.fig);

    b = OutputImage(width, height,pairs(i,2)-1,2);
    b = b(:,:,1);
    b(b ~=0) = histeq(b(b~=0));
    b = uint8(255*b);

    comb = uint8(zeros([height width 3]));
    comb(:,:,1) = base{pairs(i,1)}.v;
    comb(:,:,2) = b;

    subplot(2,1,1); imshow(b);
    subplot(2,1,2); imshow(base{pairs(i,1)}.v);
    %subplot(3,1,3); imshow(comb);

    drawnow
    fprintf('current transform:\n     metric = %1.3f\n     translation = [%1.2f, %1.2f, %1.2f]\n     rotation = [%1.2f, %1.2f, %1.2f]\n\n',...
        f(k),tform(1),tform(2),tform(3),(180*tform(4)/pi),(180*tform(5)/pi),(180*tform(6)/pi));

    FIG.count = FIG.count+1;
    %print(FIG.fig,'-dpng',['out' num2str(FIG.count) '.png']);
    %print(h,'-dpng',['plot' num2str(FIG.count) '.png']);
    sfigure(h);
end