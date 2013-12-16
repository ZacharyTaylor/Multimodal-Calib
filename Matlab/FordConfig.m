function [ config ] = FordConfig()
%FORDCONFIG Sets up struct containg setup for ford dataset

config = struct;

config.T = [0.3, 0, -0.4, 0, 0, pi/2];

% Set up cameras
config.cam{1}.K = [408.397136 0 806.586960 0;...
    0 (408.397136/2) 315.535008 0;...
    0 0 1 0];
config.cam{1}.T = [0.042152, -0.001818, -0.000285, 172.292248, 89.796352, 172.153363];
config.cam{1}.Path = '/Cam0/';
config.cam{1}.Mask = '/Mask/Cam0.png';

config.cam{2}.K = [402.206240 0 784.646528 0;...
    0 (402.206240/2) 312.174112 0;...
    0 0 1 0];
config.cam{2}.T = [0.011077, -0.040167, 0.000021, 29.509045, 89.733556, -42.675558];
config.cam{2}.Path = '/Cam1/';
config.cam{2}.Mask = '/Mask/Cam1.png';

config.cam{3}.K = [398.799712 0 818.201152 0;...
    0 (398.799712/2) 314.665832 0;...
    0 0 1 0];
config.cam{3}.T = [-0.034641, -0.023357, 0.000269, 1.344044, 89.791940, -142.573042]; 
config.cam{3}.Path = '/Cam2/';
config.cam{3}.Mask = '/Mask/Cam2.png';

config.cam{4}.K = [406.131504 0 820.718880 0;...
    0 (406.131504/2) 311.271768 0;...
    0 0 1 0];
config.cam{4}.T = [-0.033133, 0.025897, -0.000102, 152.430265, 89.517379, -63.636589];
config.cam{4}.Path = '/Cam3/';
config.cam{4}.Mask = '/Mask/Cam3.png';

config.cam{5}.K = [400.730832 0 796.724512 0;...
    0 (400.730832/2) 309.057248 0;...
    0 0 1 0];
config.cam{5}.T = [0.014544, 0.039445, 0.000097, -151.239716, 89.507811, -79.233073];
config.cam{5}.Path = '/Cam4/';
config.cam{5}.Mask = '/Mask/Cam4.png';

%convert Transforms to matricies
for i = 1:5
    config.cam{i}.T(4:6) = pi*config.cam{i}.T(4:6)/180;
    config.cam{i}.T = CreateTformMat(config.cam{i}.T);
end

end

