figure;

subplot(1,2,1); 
hold on;
plot(x,val(:,1),'b-','LineWidth',2);
plot(x,val(:,2),'r-.','LineWidth',2);
plot(x,val(:,8),'g--','LineWidth',2);
title('Offset in X')
legend('0m','1m','10m')

% subplot(2,3,2); 
% hold on;
% plot(x,val(:,1),'b-','LineWidth',2);
% plot(x,val(:,3),'r:','LineWidth',2);
% plot(x,val(:,9),'g--','LineWidth',2);
% title('Offset in Y')
% legend('0m','1m','10m')
% 
% subplot(2,3,3); 
% hold on;
% plot(x,val(:,1),'b-','LineWidth',2);
% plot(x,val(:,4),'r:','LineWidth',2);
% plot(x,val(:,10),'g--','LineWidth',2);
% title('Offset in Z')
% legend('0m','1m','10m')
% 
% subplot(2,3,4); 
% hold on;
% plot(x,val(:,1),'b-','LineWidth',2);
% plot(x,val(:,5),'r:','LineWidth',2);
% plot(x,val(:,11),'g--','LineWidth',2);
% title('Offset in Pitch')
% legend('0 degrees','10 degrees','45 degrees')

subplot(1,2,2); 
hold on;
plot(x,val(:,1),'b-','LineWidth',2);
plot(x,val(:,6),'r-.','LineWidth',2);
plot(x,val(:,12),'g--','LineWidth',2);
title('Offset in Roll')
legend('0 degrees','10 degrees','45 degrees')

% subplot(2,3,6); 
% hold on;
% plot(x,val(:,1),'b-','LineWidth',2);
% plot(x,val(:,7),'r:','LineWidth',2);
% plot(x,val(:,13),'g--','LineWidth',2);
% title('Offset in Yaw')
% legend('0 degrees','10 degrees','45 degrees')