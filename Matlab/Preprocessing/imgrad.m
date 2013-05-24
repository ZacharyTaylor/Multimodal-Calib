function [ Mag, phase ] = imgrad( I )
%IMGRAD Summary of this function goes here
%   Detailed explanation goes here

si = 50;

sigmaV = sqrt(size(I,1)/200)
sigmaH = sqrt(size(I,2)/200)

v = fspecial( 'gaussian', [si 1], sigmaV ); % vertical filter
h = fspecial( 'gaussian', [1 si], sigmaH ); % horizontal
f = v * h;

I=im2double(I);
I=imfilter(I,f,'conv');

%% Compute Gaussian derivatives

 x=-si:1:si;
 y=-si:1:si;
gaussx=-(x/(sigmaH*sigmaH)).*exp(-(x.*x+y.*y)/(2*sigmaH*sigmaH));
gaussy=(-(y/(sigmaV*sigmaV)).*exp(-(y.*y+x.*x)/(2*sigmaV*sigmaV)))';
Ix=imfilter(I,gaussx,'conv');
Iy=imfilter(I,gaussy,'conv');
%% Compute magnitude and orientation of gradient vector.

Mag=sqrt(Ix .^ 2 + Iy .^ 2);
Magmax=max(Mag(:));
Mag=Mag/Magmax;

phase = atan2d(Iy,Ix);
end
