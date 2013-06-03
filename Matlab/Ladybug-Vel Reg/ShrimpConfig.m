function [ Shrimp ] = ShrimpConfig()
%SHRIMPCONFIG Config information for shrimp

%% ladybug
Shrimp.Ladybug.offset = [0.0064,0,-1.2508,3.14159,0,0];

Shrimp.Ladybug.cam0.offset = [0.042,-0.002,0.0,-1.575,-0.004,-1.573];
Shrimp.Ladybug.cam0.focal = 403.431;
Shrimp.Ladybug.cam0.centre = [621,811.810];

Shrimp.Ladybug.cam1.offset = [0.011,-0.04,0.0,-1.571,0.001,-2.832];
Shrimp.Ladybug.cam1.focal = 410.810;
Shrimp.Ladybug.cam1.centre = [623,810.002];

Shrimp.Ladybug.cam2.offset = [-0.035,-0.023,0.0,-1.571,0.005,2.197];
Shrimp.Ladybug.cam2.focal = 412.206;
Shrimp.Ladybug.cam2.centre = [628,810.771];

Shrimp.Ladybug.cam3.offset = [-0.033,0.026,0.0,-1.572,0.004,0.94];
Shrimp.Ladybug.cam3.focal = 409.433;
Shrimp.Ladybug.cam3.centre = [609,816.045];

Shrimp.Ladybug.cam4.offset = [0.015,0.039,0.0,-1.577,-0.003,-0.317];
Shrimp.Ladybug.cam4.focal = 404.472;
Shrimp.Ladybug.cam4.centre = [627,826.407];

Shrimp.Ladybug.cam5.offset = [0.001,-0.001,0.062,0.002,0.003,-1.57];
Shrimp.Ladybug.cam5.focal = 408.488;
Shrimp.Ladybug.cam5.centre = [617,802.263];

%% velodyne
Shrimp.Velodyne.offset = [0.0215,0.0026,-0.8822,-0.0014,0.0087,3.1143];
end

