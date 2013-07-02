function [ Ladybug ] = FordConfig()
%LADYBUGCONFIG Config information for ladybug camera mounted to shrimp

Ladybug.offset = [0 0.3 -0.4 0 0 -90] + [0.5 0.5 0.5 3 3 3]/3;
Ladybug.offset(4:6) = pi*Ladybug.offset(4:6)/180;

Ladybug.cam0.offset = [0.042152, -0.001818, -0.000285, 172.292248, 89.796352, 172.153363];
Ladybug.cam0.offset(4:6) = pi*Ladybug.cam0.offset(4:6)/180;
Ladybug.cam0.focal = 408.397136;
Ladybug.cam0.centre = [806.586960,315.535008];

Ladybug.cam1.offset = [0.011077, -0.040167, 0.000021, 29.509045, 89.733556, -42.675558]; 
Ladybug.cam1.offset(4:6) = pi*Ladybug.cam1.offset(4:6)/180;
Ladybug.cam1.focal = 402.206240;
Ladybug.cam1.centre = [784.646528,312.174112];

Ladybug.cam2.offset = [-0.034641, -0.023357, 0.000269, 1.344044, 89.791940, -142.573042]; 
Ladybug.cam2.offset(4:6) = pi*Ladybug.cam2.offset(4:6)/180;
Ladybug.cam2.focal = 398.799712;
Ladybug.cam2.centre = [818.201152,314.665832];

Ladybug.cam3.offset = [-0.033133, 0.025897, -0.000102, 152.430265, 89.517379, -63.636589]; 
Ladybug.cam3.offset(4:6) = pi*Ladybug.cam3.offset(4:6)/180;
Ladybug.cam3.focal = 406.131504;
Ladybug.cam3.centre = [820.718880,311.271768];

Ladybug.cam4.offset = [0.014544, 0.039445, 0.000097, -151.239716, 89.507811, -79.233073]; 
Ladybug.cam4.offset(4:6) = pi*Ladybug.cam4.offset(4:6)/180;
Ladybug.cam4.focal = 400.730832;
Ladybug.cam4.centre = [796.724512,309.057248];


end

