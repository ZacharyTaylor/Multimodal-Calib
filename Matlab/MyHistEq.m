function [ out ] = MyHistEq( in )
%MYHISTEQ 

in = double(in);
in = in - min(in(:));

nz = find(in);

in(nz) = in(nz) - min(in(nz));
in(nz) = in(nz) / max(in(nz));

[~,temp] = sort(in(nz));

out = (0:1/(size(in(nz),1)-1):1)';
in(nz(temp)) = out;

out = in;





end

