function f=alignImages(base, move, tform)

    global FIG;
    FIG.count = FIG.count + 1;
    
    width = size(move,2);
    height = size(move,1);
    
    tform = double(tform);
    
    rot = [cosd(tform(3)),-sind(tform(3)),0;sind(tform(3)),cosd(tform(3)),0;0,0,1]; 
    scale = [tform(4),0,0;0,tform(5),0;0,0,1];
    shear = [1,tform(6),0;tform(7),1,0;0,0,1];
    trans = [1,0,tform(1);0,1,tform(2);0,0,1];

    tformMat = single(trans*shear*scale*rot);
    
    SetTformMatrix(tformMat);
    Transform(0);
    InterpolateBaseValues(0);
    
    f = EvalMetric(0);
    if(isnan(f) || isinf(f))
        f = 0;
    end
    
    %display current estimate
%     if(FIG.countMax < FIG.count)
%         FIG.count = 0;
%         h = gcf;
%         sfigure(FIG.fig);
%         
%         mapAffine = affineTransform([0 0 0 1 1 0 0],width,height);
%         move = outputImage(width,height,mapAffine,moveD,width*height);
%         clearMap(mapAffine);
%         
%         comb = zeros([size(move) 3]);
%         comb(:,:,1) = base;
%         comb(:,:,2) = move;
%         
%         subplot(2,2,1); imshow(move);
%         subplot(2,2,2); imshow(base);
%         subplot(2,2,3); imshow((base+move)/2);
%         subplot(2,2,4); imshow(comb);
%         
%         drawnow
%         tform
%         
%         sfigure(h);
%     end
    
    clearMap(map);
    clearImage(moveD);
    
end