clc
clear
close all
sub_image_size = 28;
% Display grey-scale image
im0_train = imread('mnist_train0.jpg');
im1_train = imread('mnist_train1.jpg');
im2_train = imread('mnist_train2.jpg');
im0_test = imread('mnist_test0.jpg');
im1_test = imread('mnist_test1.jpg');
im2_test = imread('mnist_test2.jpg');
% Convert images to binary images by threshold
bw0_train=im0_train>30;
bw1_train=im1_train>30;
bw2_train=im2_train>29;
bw0_test=im0_test>30;
bw1_test=im1_test>30;
bw2_test=im2_test>29;
%% Determine feature statistics
% Pre-define Circularity statistic arrays
circ_stat0_train = [];
circ_stat2_train = [];
circ_stat0_test = [];
circ_stat2_test = [];
% Pre-define Perimeter statistic arrays
peri_stat0_train = [];
peri_stat2_train = [];
peri_stat0_test = [];
peri_stat2_test = [];
% Pre-define Area statistic arrays
area_stat0_train = [];
area_stat1_train = [];
area_stat0_test = [];
area_stat1_test = [];
% Pre-define Eccentricity statistic arrays
ecce_stat0_train = [];
ecce_stat1_train = [];
ecce_stat0_test = [];
ecce_stat1_test = [];
% Determine feature measurements for training '0'
[rows,cols] = size(bw0_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img0 = bw0_train(row_pix,col_pix);
       if (sum(sum(sub_img0)) ~= 0)
           circ_stat = regionprops(sub_img0,'Circularity');
           peri_stat = regionprops(sub_img0,'Perimeter');
           area_stat = regionprops(sub_img0,'Area');
           ecce_stat = regionprops(sub_img0,'Eccentricity');
           % Circularity
           if (length(circ_stat) == 1)
               circ_stat0_train = [circ_stat0_train,circ_stat];
           end
           % Perimeter
           if (length(peri_stat) == 1)
               peri_stat0_train = [peri_stat0_train,peri_stat];
           end
           % Area
           if (length(area_stat) == 1)
               area_stat0_train = [area_stat0_train,area_stat];
           end
           % Eccentricity
           if (length(ecce_stat) == 1)
               ecce_stat0_train = [ecce_stat0_train,ecce_stat];
           end
       end
   end
end
% Determine feature measurements for testing '0'
[rows,cols] = size(bw0_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img0 = bw0_test(row_pix,col_pix);
       if (sum(sum(sub_img0)) ~= 0)
           circ_stat = regionprops(sub_img0,'Circularity');
           peri_stat = regionprops(sub_img0,'Perimeter');
           area_stat = regionprops(sub_img0,'Area');
           ecce_stat = regionprops(sub_img0,'Eccentricity');
           % Circularity
           if (length(circ_stat) == 1)
               circ_stat0_test = [circ_stat0_test,circ_stat];
           end
           % Perimeter
           if (length(peri_stat) == 1)
               peri_stat0_test = [peri_stat0_test,peri_stat];
           end
           % Area
           if (length(area_stat) == 1)
               area_stat0_test = [area_stat0_test,area_stat];
           end
           % Eccentricity
           if (length(ecce_stat) == 1)
               ecce_stat0_test = [ecce_stat0_test,ecce_stat];
           end
       end
   end
end
% Determine feature measurements for '1'
[rows,cols] = size(bw1_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img1 = bw1_train(row_pix,col_pix);
       if (sum(sum(sub_img1)) ~= 0)
           area_stat = regionprops(sub_img1,'Area');
           ecce_stat = regionprops(sub_img1,'Eccentricity');
           % Area
           if (length(area_stat) == 1)
               area_stat1_train = [area_stat1_train,area_stat];
           end
           % Eccentricity
           if (length(ecce_stat) == 1)
               ecce_stat1_train = [ecce_stat1_train,ecce_stat];
           end
       end
   end
end
% Determine feature measurements for testing '1'
[rows,cols] = size(bw1_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img1 = bw1_test(row_pix,col_pix);
       if (sum(sum(sub_img1)) ~= 0)
           area_stat = regionprops(sub_img1,'Area');
           ecce_stat = regionprops(sub_img1,'Eccentricity');
           % Area
           if (length(area_stat) == 1)
               area_stat1_test = [area_stat1_test,area_stat];
           end
           % Eccentricity
           if (length(ecce_stat) == 1)
               ecce_stat1_test = [ecce_stat1_test,ecce_stat];
           end
       end
   end
end
% Determine feature measurements for '2'
[rows,cols] = size(bw2_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img2 = bw2_train(row_pix,col_pix);
       if (sum(sum(sub_img2)) ~= 0)
           circ_stat = regionprops(sub_img2,'Circularity');
           peri_stat = regionprops(sub_img2,'Perimeter');
           % Circularity
           if (length(circ_stat) == 1)
               circ_stat2_train = [circ_stat2_train,circ_stat];
           end
           % Perimeter
           if (length(peri_stat) == 1)
               peri_stat2_train = [peri_stat2_train,peri_stat];
           end
       end
   end
end
% Determine feature measurements for testing '2'
[rows,cols] = size(bw2_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img2 = bw2_test(row_pix,col_pix);
       if (sum(sum(sub_img2)) ~= 0)
           circ_stat = regionprops(sub_img2,'Circularity');
           peri_stat = regionprops(sub_img2,'Perimeter');
           % Circularity
           if (length(circ_stat) == 1)
               circ_stat2_test = [circ_stat2_test,circ_stat];
           end
           % Perimeter
           if (length(peri_stat) == 1)
               peri_stat2_test = [peri_stat2_test,peri_stat];
           end
       end
   end
end
%% Create Input Arrays
Inputs_0 = ones(3,length([ecce_stat0_train.Eccentricity]));
Inputs_0(2,:) = [ecce_stat0_train.Eccentricity];
Inputs_0(3,:) = [area_stat0_train.Area];
Inputs_1 = -1*ones(3,length([ecce_stat1_train.Eccentricity]));
Inputs_1(2,:) = -1*[ecce_stat1_train.Eccentricity];
Inputs_1(3,:) = -1*[area_stat1_train.Area];
Inputs_0v1 = [Inputs_0,Inputs_1];
Inputs_0 = ones(3,length([circ_stat0_train.Circularity]));
Inputs_0(2,:) = [circ_stat0_train.Circularity];
Inputs_0(3,:) = [peri_stat0_train.Perimeter];
Inputs_2 = -1*ones(3,length([circ_stat2_train.Circularity]));
Inputs_2(2,:) = -1*[circ_stat2_train.Circularity];
Inputs_2(3,:) = -1*[peri_stat2_train.Perimeter];
Inputs_0v2 = [Inputs_0,Inputs_2];
Inputs_0_test = ones(3,length([ecce_stat0_test.Eccentricity]));
Inputs_0_test(3,:) = [ecce_stat0_test.Eccentricity];
Inputs_0_test(2,:) = [area_stat0_test.Area];
Inputs_1_test = -1*ones(3,length([ecce_stat1_test.Eccentricity]));
Inputs_1_test(3,:) = -1*[ecce_stat1_test.Eccentricity];
Inputs_1_test(2,:) = -1*[area_stat1_test.Area];
Inputs_0v1_test = [Inputs_0_test,Inputs_1_test];
Inputs_0_test = ones(3,length([circ_stat0_test.Circularity]));
Inputs_0_test(2,:) = [circ_stat0_test.Circularity];
Inputs_0_test(3,:) = [peri_stat0_test.Perimeter];
Inputs_2_test = -1*ones(3,length([circ_stat2_test.Circularity]));
Inputs_2_test(2,:) = -1*[circ_stat2_test.Circularity];
Inputs_2_test(3,:) = -1*[peri_stat2_test.Perimeter];
Inputs_0v2_test = [Inputs_0_test,Inputs_2_test];
%% O vs. 1 Batch
eta = 1;
initial_weights = [1 1 1]';
w_b_01 = BatchPerceptron(Inputs_0v1,initial_weights,eta,4);
figure(1)
plot([area_stat0_train.Area],[ecce_stat0_train.Eccentricity],'.')
hold on
plot([area_stat1_train.Area],[ecce_stat1_train.Eccentricity],'.')
line([0,250],[-(w_b_01(1))/w_b_01(3),-(w_b_01(1) + 250*w_b_01(2))/w_b_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Batch Perceptron for 0 vs. 1 Training Data')
legend('0','1','Boundary')
figure(2)
plot([area_stat0_test.Area],[ecce_stat0_test.Eccentricity],'.')
hold on
plot([area_stat1_test.Area],[ecce_stat1_test.Eccentricity],'.')
line([0,250],[-(w_b_01(1))/w_b_01(3),-(w_b_01(1) + 250*w_b_01(2))/w_b_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Batch Perceptron for 0 vs. 1 Testing Data')
legend('0','1','Boundary')
%% 0 vs. 1 Single Sample
eta = 0.49875;
initial_weights = [1 1 1]';
[~,cols] = size(Inputs_0v1);
w_s_01 = SingleSamplePerceptron(Inputs_0v1,initial_weights,eta,3*cols);
figure(3)
plot([area_stat0_train.Area],[ecce_stat0_train.Eccentricity],'.')
hold on
plot([area_stat1_train.Area],[ecce_stat1_train.Eccentricity],'.')
line([0,250],[-(w_s_01(1))/w_s_01(3),-(w_s_01(1) + 250*w_s_01(2))/w_s_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Single Sample Perceptron for 0 vs. 1 Training Data')
legend('0','1','Boundary')
figure(4)
plot([area_stat0_test.Area],[ecce_stat0_test.Eccentricity],'.')
hold on
plot([area_stat1_test.Area],[ecce_stat1_test.Eccentricity],'.')
line([0,250],[-(w_s_01(1))/w_s_01(3),-(w_s_01(1) + 250*w_s_01(2))/w_s_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Single Sample Perceptron for 0 vs. 1 Testing Data')
legend('0','1','Boundary')
%% 0 vs. 1 Ho-Kashyap
Inputs_0 = ones(3,length([ecce_stat0_train.Eccentricity]));
Inputs_0(3,:) = [ecce_stat0_train.Eccentricity];
Inputs_0(2,:) = [area_stat0_train.Area];
Inputs_1 = -1*ones(3,length([ecce_stat1_train.Eccentricity]));
Inputs_1(3,:) = -1*[ecce_stat1_train.Eccentricity];
Inputs_1(2,:) = -1*[area_stat1_train.Area];
Inputs_0v1 = [Inputs_0,Inputs_1]';
eta = 0.1;
initial_weights = 1*rand([3,1]);
[rows,~] = size(Inputs_0v1);
rng(0)
initial_dists = 100*rand([rows,1]);
w_hk_01 = HoKashyapAlgo(Inputs_0v1,initial_weights,initial_dists,eta,100000);
figure(5)
plot([area_stat0_train.Area],[ecce_stat0_train.Eccentricity],'.')
hold on
plot([area_stat1_train.Area],[ecce_stat1_train.Eccentricity],'.')
line([0,250],[-(w_hk_01(1))/w_hk_01(3),-(w_hk_01(1) + 250*w_hk_01(2))/w_hk_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Ho-Kashyap for 0 vs. 1 Training Data')
legend('0','1','Boundary')
figure(6)
plot([area_stat0_test.Area],[ecce_stat0_test.Eccentricity],'.')
hold on
plot([area_stat1_test.Area],[ecce_stat1_test.Eccentricity],'.')
line([0,250],[-(w_hk_01(1))/w_hk_01(3),-(w_hk_01(1) + 250*w_hk_01(2))/w_hk_01(3)])
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('Ho-Kashyap for 0 vs. 1 Testing Data')
legend('0','1','Boundary')
%% 0 vs.1 SVM
Inputs_0 = ones(2,length([ecce_stat0_train.Eccentricity]));
Inputs_0(2,:) = [ecce_stat0_train.Eccentricity];
Inputs_0(1,:) = [area_stat0_train.Area];
Class_0 = ones(1,length([ecce_stat0_train.Eccentricity]));
Inputs_1 = ones(2,length([ecce_stat1_train.Eccentricity]));
Inputs_1(2,:) = [ecce_stat1_train.Eccentricity];
Inputs_1(1,:) = [area_stat1_train.Area];
Class_1 = -1*ones(1,length([ecce_stat1_train.Eccentricity]));
Inputs_0v1_svm = [Inputs_0,Inputs_1]';
Classes_0v1_svm = [Class_0,Class_1]';
svm_mdl_0v1 = fitcsvm(Inputs_0v1_svm,Classes_0v1_svm,'KernelFunction','linear');
sv = svm_mdl_0v1.SupportVectors;
d = 0.02;
[x1Grid,x2Grid] = meshgrid(0:d:350, ...
   0:d:1.1);
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svm_mdl_0v1,xGrid);
figure(7)
plot([area_stat0_train.Area],[ecce_stat0_train.Eccentricity],'.')
hold on
plot([area_stat1_train.Area],[ecce_stat1_train.Eccentricity],'.')
plot(sv(:,1),sv(:,2),'ko','MarkerSize',5)
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0], ...
   'LineColor',[0, 0.4470, 0.7410]);
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('SVM for 0 vs. 1 Training Data')
legend('0','1','Supp. Vect.','Boundary')
figure(8)
plot([area_stat0_test.Area],[ecce_stat0_test.Eccentricity],'.')
hold on
plot([area_stat1_test.Area],[ecce_stat1_test.Eccentricity],'.')
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0], ...
   'LineColor',[0, 0.4470, 0.7410]);
xlim([0,350])
ylim([0,1.1])
xlabel('Area')
ylabel('Eccentricity')
title('SVM for 0 vs. 1 Testing Data')
legend('0','1','Boundary')
%% O vs. 2 Batch
eta = 1;
initial_weights = [1 1 1]';
w_b_02 = BatchPerceptron(Inputs_0v2,initial_weights,eta,100000);
figure(9)
plot([circ_stat0_train.Circularity],[peri_stat0_train.Perimeter],'.')
hold on
plot([circ_stat2_train.Circularity],[peri_stat2_train.Perimeter],'.')
line([0,1.1],[-(w_b_02(1))/w_b_02(3),-(w_b_02(1) + 1.1*w_b_02(2))/w_b_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Batch Perceptron for 0 vs. 2 Training Data')
legend('0','2','Boundary')
figure(10)
plot([circ_stat0_test.Circularity],[peri_stat0_test.Perimeter],'.')
hold on
plot([circ_stat2_test.Circularity],[peri_stat2_test.Perimeter],'.')
line([0,1.1],[-(w_b_02(1))/w_b_02(3),-(w_b_02(1) + 1.1*w_b_02(2))/w_b_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Batch Perceptron for 0 vs. 2 Testing Data')
legend('0','2','Boundary')
%% O vs. 2 Single Sample
eta = 0.6;
initial_weights = [1 1 1]';
[~,cols] = size(Inputs_0v2);
w_s_02 = SingleSamplePerceptron(Inputs_0v2,initial_weights,eta,100000*cols);
figure(11)
plot([circ_stat0_train.Circularity],[peri_stat0_train.Perimeter],'.')
hold on
plot([circ_stat2_train.Circularity],[peri_stat2_train.Perimeter],'.')
line([0,1.1],[-(w_s_02(1))/w_s_02(3),-(w_s_02(1) + 1.1*w_s_02(2))/w_s_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Single Sample Perceptron for 0 vs. 2 Training Data')
legend('0','2','Boundary')
figure(12)
plot([circ_stat0_test.Circularity],[peri_stat0_test.Perimeter],'.')
hold on
plot([circ_stat2_test.Circularity],[peri_stat2_test.Perimeter],'.')
line([0,1.1],[-(w_s_02(1))/w_s_02(3),-(w_s_02(1) + 1.1*w_s_02(2))/w_s_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Single Sample Perceptron for 0 vs. 2 Testing Data')
legend('0','2','Boundary')
%% 0 vs. 2 Ho-Kashyap
eta = 0.0001;
initial_weights = [1 1 1]';
[rows,~] = size(Inputs_0v2');
rng(0)
initial_dists = 100*rand([rows,1]);
w_hk_02 = HoKashyapAlgo(Inputs_0v2',initial_weights,initial_dists,eta,1);
figure(13)
plot([circ_stat0_train.Circularity],[peri_stat0_train.Perimeter],'.')
hold on
plot([circ_stat2_train.Circularity],[peri_stat2_train.Perimeter],'.')
line([0,1.1],[-(w_hk_02(1))/w_hk_02(3),-(w_hk_02(1) + 1.1*w_hk_02(2))/w_hk_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Ho-Kashyap for 0 vs. 2 Training Data')
legend('0','2','Boundary')
figure(14)
plot([circ_stat0_test.Circularity],[peri_stat0_test.Perimeter],'.')
hold on
plot([circ_stat2_test.Circularity],[peri_stat2_test.Perimeter],'.')
line([0,1.1],[-(w_hk_02(1))/w_hk_02(3),-(w_hk_02(1) + 1.1*w_hk_02(2))/w_hk_02(3)])
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Ho-Kashyap for 0 vs. 2 Testing Data')
legend('0','2','Boundary')
%% 0 vs.2 SVM
Inputs_0 = ones(2,length([circ_stat0_train.Circularity]));
Inputs_0(1,:) = [circ_stat0_train.Circularity];
Inputs_0(2,:) = [peri_stat0_train.Perimeter];
Class_0 = ones(1,length([circ_stat0_train.Circularity]));
Inputs_2 = ones(2,length([circ_stat2_train.Circularity]));
Inputs_2(1,:) = [circ_stat2_train.Circularity];
Inputs_2(2,:) = [peri_stat2_train.Perimeter];
Class_2 = -1*ones(1,length([circ_stat2_train.Circularity]));
Inputs_0v2_svm = [Inputs_0,Inputs_2]';
Classes_0v2_svm = [Class_0,Class_2]';
svm_mdl_0v2 = fitcsvm(Inputs_0v2_svm,Classes_0v2_svm,'KernelFunction','linear');
sv = svm_mdl_0v2.SupportVectors;
d = 0.02;
[x1Grid,x2Grid] = meshgrid(0:d:1.1,0:d:150);
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svm_mdl_0v2,xGrid);
figure(15)
plot([circ_stat0_train.Circularity],[peri_stat0_train.Perimeter],'.')
hold on
plot([circ_stat2_train.Circularity],[peri_stat2_train.Perimeter],'.')
plot(sv(:,1),sv(:,2),'ko','MarkerSize',5)
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0], ...
   'LineColor',[0, 0.4470, 0.7410]);
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('Ho-Kashyap for 0 vs. 2 Training Data')
legend('0','1','Supp. Vect.','Boundary')
figure(16)
plot([circ_stat0_test.Circularity],[peri_stat0_test.Perimeter],'.')
hold on
plot([circ_stat2_test.Circularity],[peri_stat2_test.Perimeter],'.')
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0], ...
   'LineColor',[0, 0.4470, 0.7410]);
xlim([0,1.1])
ylim([0,150])
xlabel('Circularity')
ylabel('Perimeter')
title('SVM for 0 vs. 2 Testing Data')
legend('0','1','Boundary')
%% Extract SVM Raw Pixels
% Extract pixel information for training '0'
Input_0_svm_raw = [];
[rows,cols] = size(im0_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img0 = im0_train(row_pix,col_pix);
       if (sum(sum(sub_img0)) ~= 0)
           Input_0_svm_raw = [Input_0_svm_raw;reshape(sub_img0',1,[])];
       end
   end
end
% Extract pixel information for testing '0'
Input_0_svm_raw_test = [];
[rows,cols] = size(im0_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img0 = im0_test(row_pix,col_pix);
       if (sum(sum(sub_img0)) ~= 0)
           Input_0_svm_raw_test = [Input_0_svm_raw_test;reshape(sub_img0',1,[])];
       end
   end
end
% Extract pixel information for training '1'
Input_1_svm_raw = [];
[rows,cols] = size(im1_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img1 = im1_train(row_pix,col_pix);
       if (sum(sum(sub_img1)) ~= 0)
           Input_1_svm_raw = [Input_1_svm_raw;reshape(sub_img1',1,[])];
       end
   end
end
% Extract pixel information for testing '1'
Input_1_svm_raw_test = [];
[rows,cols] = size(im1_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img1 = im1_test(row_pix,col_pix);
       if (sum(sum(sub_img1)) ~= 0)
           Input_1_svm_raw_test = [Input_1_svm_raw_test;reshape(sub_img1',1,[])];
       end
   end
end
% Extract pixel information for training '2'
Input_2_svm_raw = [];
[rows,cols] = size(im2_train);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img2 = im2_train(row_pix,col_pix);
       if (sum(sum(sub_img2)) ~= 0)
           Input_2_svm_raw = [Input_2_svm_raw;reshape(sub_img2',1,[])];
       end
   end
end
% Extract pixel information for testing '2'
Input_2_svm_raw_test = [];
[rows,cols] = size(im2_test);
for m = 1:sub_image_size:rows
   for n = 1:sub_image_size:cols
       row_pix = m:m + sub_image_size - 1;
       col_pix = n:n + sub_image_size - 1;
       sub_img2 = im2_test(row_pix,col_pix);
       if (sum(sum(sub_img2)) ~= 0)
           Input_2_svm_raw_test = [Input_2_svm_raw_test;reshape(sub_img2',1,[])];
       end
   end
end
% 0 vs. 1 SVM Raw Pixels
Inputs_0v1_svm_raw = cast([Input_0_svm_raw',Input_1_svm_raw']','double');
Classes_0v1_svm_raw = [ones(1,size(Input_0_svm_raw,1)),...
   -1*ones(1,size(Input_1_svm_raw,1))]';
svm_mdl_0v1_raw = fitcsvm(Inputs_0v1_svm_raw,Classes_0v1_svm_raw, ...
   "KernelFunction","linear");
% 0 vs. 2 SVM Raw Pixels
Inputs_0v2_svm_raw = cast([Input_0_svm_raw',Input_2_svm_raw']','double');
Classes_0v2_svm_raw = [ones(1,size(Input_0_svm_raw,1)),...
   -1*ones(1,size(Input_2_svm_raw,1))]';
svm_mdl_0v2_raw = fitcsvm(Inputs_0v2_svm_raw,Classes_0v2_svm_raw, ...
   "KernelFunction","linear");
%% Calculate Error Rates
% Batch 0v1
y_batch_0v1 = w_b_01'*-1*Inputs_0v1_test;
y_batch_0v1_miss = 0;
for i = 1:length(y_batch_0v1)
   if (y_batch_0v1(i) <= 0)
       y_batch_0v1_miss = y_batch_0v1_miss + 1;
   end
end
y_batch_0v1_error_rate = y_batch_0v1_miss/size(Inputs_0v1_test,2)*100
% Batch 0v2
y_batch_0v2 = w_b_02'*Inputs_0v2_test;
y_batch_0v2_miss = 0;
for i = 1:length(y_batch_0v2)
   if (y_batch_0v2(i) <= 0)
       y_batch_0v2_miss = y_batch_0v2_miss + 1;
   end
end
y_batch_0v2_error_rate = y_batch_0v2_miss/size(Inputs_0v2_test,2)*100
% Single 0v1
y_single_0v1 = w_s_01'*-1*Inputs_0v1_test;
y_single_0v1_miss = 0;
for i = 1:length(y_single_0v1)
   if (y_single_0v1(i) <= 0)
       y_single_0v1_miss = y_single_0v1_miss + 1;
   end
end
y_single_0v1_error_rate = y_single_0v1_miss/size(Inputs_0v1_test,2)*100
% Single 0v2
y_single_0v2 = w_s_02'*Inputs_0v2_test;
y_single_0v2_miss = 0;
for i = 1:length(y_single_0v2)
   if (y_single_0v2(i) <= 0)
       y_single_0v2_miss = y_single_0v2_miss + 1;
   end
end
y_single_0v2_error_rate = y_single_0v2_miss/size(Inputs_0v2_test,2)*100
% Ho-Kashyap 0v1
y_ho_kash_0v1 = w_hk_01'*Inputs_0v1_test;
y_ho_kash_0v1_miss = 0;
for i = 1:length(y_ho_kash_0v1)
   if (y_ho_kash_0v1(i) <= 0)
       y_ho_kash_0v1_miss = y_ho_kash_0v1_miss + 1;
   end
end
y_ho_kash_0v1_error_rate = y_ho_kash_0v1_miss/size(Inputs_0v1_test,2)*100
% Ho-Kashyap 0v2
y_ho_kash_0v2 = w_hk_02'*Inputs_0v2_test;
y_ho_kash_0v2_miss = 0;
for i = 1:length(y_ho_kash_0v2)
   if (y_ho_kash_0v2(i) <= 0)
       y_ho_kash_0v2_miss = y_ho_kash_0v2_miss + 1;
   end
end
y_ho_kash_0v2_error_rate = y_ho_kash_0v2_miss/size(Inputs_0v2_test,2)*100
% SVM 0v1
Inputs_0 = ones(2,length([ecce_stat0_test.Eccentricity]));
Inputs_0(2,:) = [ecce_stat0_test.Eccentricity];
Inputs_0(1,:) = [area_stat0_test.Area];
Class_0 = ones(1,length([ecce_stat0_test.Eccentricity]));
Inputs_1 = ones(2,length([ecce_stat1_test.Eccentricity]));
Inputs_1(2,:) = [ecce_stat1_test.Eccentricity];
Inputs_1(1,:) = [area_stat1_test.Area];
Class_1 = -1*ones(1,length([ecce_stat1_test.Eccentricity]));
Inputs_0v1_svm_test = [Inputs_0,Inputs_1]';
Classes_0v1_svm_test = [Class_0,Class_1]';
pred_svm_0v1_test = predict(svm_mdl_0v1,Inputs_0v1_svm_test);
y_svm_0v1_miss = 0;
for i = 1:length(Classes_0v1_svm_test)
   if (pred_svm_0v1_test(i) ~= Classes_0v1_svm_test(i))
       y_svm_0v1_miss = y_svm_0v1_miss + 1;
   end
end
y_svm_0v1_error_rate = y_svm_0v1_miss/size(Classes_0v1_svm_test,1)*100
% SVM 0v2
Inputs_0 = ones(2,length([circ_stat0_test.Circularity]));
Inputs_0(1,:) = [circ_stat0_test.Circularity];
Inputs_0(2,:) = [peri_stat0_test.Perimeter];
Class_0 = ones(1,length([circ_stat0_test.Circularity]));
Inputs_2 = ones(2,length([circ_stat2_test.Circularity]));
Inputs_2(1,:) = [circ_stat2_test.Circularity];
Inputs_2(2,:) = [peri_stat2_test.Perimeter];
Class_2 = -1*ones(1,length([circ_stat2_test.Circularity]));
Inputs_0v2_svm_test = [Inputs_0,Inputs_2]';
Classes_0v2_svm_test = [Class_0,Class_2]';
pred_svm_0v2_test = predict(svm_mdl_0v2,Inputs_0v2_svm_test);
y_svm_0v2_miss = 0;
for i = 1:length(Classes_0v2_svm_test)
   if (pred_svm_0v2_test(i) ~= Classes_0v2_svm_test(i))
       y_svm_0v2_miss = y_svm_0v2_miss + 1;
   end
end
y_svm_0v2_error_rate = y_svm_0v2_miss/size(Classes_0v2_svm_test,1)*100
% SVM raw 0v1
Inputs_0v1_svm_raw_test = cast([Input_0_svm_raw_test',Input_1_svm_raw_test']' ...
   ,'double');
Classes_0v1_svm_raw_test = [ones(1,size(Input_0_svm_raw_test,1)),...
   -1*ones(1,size(Input_1_svm_raw_test,1))]';
pred_svm_0v1_raw = predict(svm_mdl_0v1_raw,Inputs_0v1_svm_raw_test);
y_svm_0v1_raw_miss = 0;
for i = 1:length(Classes_0v1_svm_raw_test)
   if (pred_svm_0v1_raw(i) ~= Classes_0v1_svm_raw_test(i))
       y_svm_0v1_raw_miss = y_svm_0v1_raw_miss + 1;
   end
end
y_svm_0v1_raw_error_rate = y_svm_0v1_raw_miss/size(Classes_0v1_svm_raw_test,1)*100
% SVM raw 0v2
Inputs_0v2_svm_raw_test = cast([Input_0_svm_raw_test',Input_2_svm_raw_test']' ...
   ,'double');
Classes_0v2_svm_raw_test = [ones(1,size(Input_0_svm_raw_test,1)),...
   -1*ones(1,size(Input_2_svm_raw_test,1))]';
pred_svm_0v2_raw = predict(svm_mdl_0v2_raw,Inputs_0v2_svm_raw_test);
y_svm_0v2_raw_miss = 0;
for i = 1:length(Classes_0v2_svm_raw_test)
   if (pred_svm_0v2_raw(i) ~= Classes_0v2_svm_raw_test(i))
       y_svm_0v2_raw_miss = y_svm_0v2_raw_miss + 1;
   end
end
y_svm_0v2_raw_error_rate = y_svm_0v2_raw_miss/size(Classes_0v2_svm_raw_test,1)*100



BatchPerceptron.m

function weights = BatchPerceptron(inputs,weights,eta,iterations)
%BatchPerceptron: Function is an implementation of the Batch Perceptron
%algorithm. Given some initial parameters, the algorithm will determine a
%seperating boundary suitable for the inputs. Feature space can be of size
%greater than or equal to 1.
%
%   Parameters:
%       inputs - Initial input values (normalized & augmented)
%       weights - Initial weight values (normalized & augmented)
%       dists - Initial distances from inputs to decision boundary
%       eta - Learning rate between 0 and 1
%       iterations - Number of times to run the algorithm before concluding
%  
%   Outputs:
%       weights - Final value of weights

   % Run the algorithm as many times specified by iterations
   [rows,cols] = size(inputs);
   for k = 1:iterations
       sum_misclass = zeros(rows,1); % Reset sum
       for i = 1:cols
           if (~(weights' * inputs(:,i) > 0)) % Check classification
               % Add misclassifications
               sum_misclass = sum_misclass + inputs(:,i);
           end
       end
       weights = weights + eta*sum_misclass; % Update weights
   end
end



SingleSamplePerceptron.m

function weights = SingleSamplePerceptron(inputs,weights,eta,iterations)
%SingleSamplePerceptron: Function is an implementation of the Single Sample
%Perceptron algorithm. Given some initial parameters, the algorithm will
%determine a seperating boundary suitable for the inputs. Feature space
%can be of size greater than or equal to 1.
%
%   Parameters:
%       inputs - Initial input values (normalized & augmented)
%       weights - Initial weight values (normalized & augmented)
%       dists - Initial distances from inputs to decision boundary
%       eta - Learning rate between 0 and 1
%       iterations - Number of times to run the algorithm before concluding
%  
%   Outputs:
%       weights - Final value of weights

   % Run the algorithm as many times specified by iterations
   [~,cols] = size(inputs);
   for k = 0:iterations
       % Wrap around to first sample at end
       current_index = mod(k,cols) + 1;
       if (~(weights' * inputs(:,current_index) > 0)) % Check classification
           weights = weights + eta*inputs(:,current_index); % Adjust weight
       end
   end
end



HoKashyapAlgo.m

function [weights,dists] = HoKashyapAlgo(inputs,weights,dists,eta,iterations)
%HoKashyapAlgo: Function is an implementation of the Ho-Kashyap algorithm.
%Given some initial parameters, the algorithm will determine a seperating
%boundary suitable for the inputs. Feature space can be of size greater
%than or equal to 1.
%
%   Parameters:
%       inputs - Initial input values (normalized & augmented)
%       weights - Initial weight values (normalized & augmented)
%       dists - Initial distances from inputs to decision boundary
%       eta - Learning rate between 0 and 1
%       iterations - Number of times to run the algorithm before concluding
%  
%   Outputs:
%       weights - Final value of weights
%       dists - Final value of distances

   % Run the algorithm as many times specified by iterations
   for i = 1:iterations
       error = inputs*weights - dists;                 % Calculate error
       dists = dists + eta*(error + abs(error));       % Update distances
       weights = ((inputs'*inputs)^-1)*inputs'*dists;  % Update weights
   end
end

