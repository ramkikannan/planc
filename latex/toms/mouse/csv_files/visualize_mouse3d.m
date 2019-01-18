% A script for visualizing the Mouse data

% Set the parameters 
k = 12;
alg = 'abpp';

% Hard coded for reading in the 3 modes.
% ******************************************
mode = '0';
L = ['mouse_output_',alg,'_',num2str(k),'_mode',mode,'_64','.csv'];
file_name = join(L);
M0 = dlmread(file_name);

mode = '1';
L = ['mouse_output_',alg,'_',num2str(k),'_mode',mode,'_64','.csv'];
file_name = join(L);
M1 = dlmread(file_name);

mode = '2';
L = ['mouse_output_',alg,'_',num2str(k),'_mode',mode,'_64','.csv'];
file_name = join(L);
M2 = dlmread(file_name);

mode = '3';
L = ['mouse_output_',alg,'_',num2str(k),'_mode',mode,'_64','.csv'];
file_name = join(L);
M3 = dlmread(file_name);
% ******************************************

% figure parameters
m = 4;
n = 1;

for l = 1:k
    figure;
    hold on;
    T = ['Mouse 3d, NNCP via ',alg,' with rank = ',num2str(k),'; component ',num2str(l)];
    sgtitle(T);
    subplot(m,n,1);
    plot(1:size(M0,1),M0(:,l)); % plot 1st mode rank l
    subplot(m,n,2);
    plot(1:size(M1,1),M1(:,l)); % plot 2nd mode rank l
    subplot(m,n,3);
    plot(1:size(M2,1),M2(:,l)); % plot 3rd mode rank l
    subplot(m,n,4);
    plot(1:size(M3,1),M3(:,l)); % plot 4th mode rank l
    
    S = ['mouse_3d_output_',alg,'_',num2str(k),'_','rank',num2str(l)];
    saveas(gcf,S,'png')
    hold off;
end
