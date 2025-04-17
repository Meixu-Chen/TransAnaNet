% clear,clc,close all

ct_path= '../CT_mask';
w1_path = '../W1_CBCT_mask';
w3_path = '../W3_CBCT_mask';
pred_path = 'Pred_W3_better';
% pred_path = 'pred_val';
nii_input_path= '../../input';
nii_label_path = '../../label';
npy_pred_path = 'pred_val_better';

index = readtable('data_loader_validation.txt');
index = table2cell(index);

dice_ct_3 = [];
dice_1_3 = [];
dice_pred_3 = [];

dice_gtv_ct_3 = [];
dice_gtv_1_3 = [];
dice_gtv_pred_3 = [];

dice_gtvn_ct_3 = [];
dice_gtvn_1_3 = [];
dice_gtvn_pred_3 = [];

asd_ct_3 = [];
asd_1_3 = [];
asd_pred_3 = [];

asd_gtv_ct_3 = [];
asd_gtv_1_3 = [];
asd_gtv_pred_3 = [];

asd_gtvn_ct_3 = [];
asd_gtvn_1_3 = [];
asd_gtvn_pred_3 = [];

mse_ct_w3=[];
mse_w1_w3=[];
mse_pred_w3=[];

psnr_ct_w3=[];
psnr_w1_w3=[];
psnr_pred_w3=[];

ssim_ct_w3=[];
ssim_w1_w3=[];
ssim_pred_w3=[];

vol_ct = [];
vol_w1 = [];
vol_pre_cnn = [];
vol_w3 = [];
vol_ct_gtv = [];
vol_ct_gtvn = [];
vol_w1_gtv = [];
vol_w1_gtvn = [];
vol_pre_gtv_cnn = [];
vol_pre_gtvn_cnn = [];
vol_w3_gtv = [];
vol_w3_gtvn = [];

% regionprop for measuring center of mass difference and max-diameter
ct_gtv_prop = [];
ct_gtvn_prop = [];
w1_gtv_prop = [];
w1_gtvn_prop = [];
w3_gtv_prop = [];
w3_gtvn_prop = [];
pre_gtv_prop = [];
pre_gtvn_prop = [];
% stats = regionprops3(cube_in_cube,'all')


for id = 0:length(index)-1
    % id = 3;


    % files = dir(fullfile(ct_path,'*.npy'));
    % ct_filename = files(id+1).name;
    ct_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CT.npy'];

    % files = dir(fullfile(w1_path,'*.npy'));
    % cbct1_filename = files(id+1).name;
    cbct1_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CBCT1.npy'];

    % files = dir(fullfile(w3_path,'*.npy'));
    % cbct3_filename = files(id+1).name;
    cbct3_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CBCT2.npy'];

    pred_file = fullfile(pred_path,['pred_img_',num2str(id),'.npy']);

    nii_ct_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CT.nii.gz']);
    nii_cbct1_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CBCT1.nii.gz']);
    nii_cbct3_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CBCT2.nii.gz']);
    pred_cbct3_File = fullfile(npy_pred_path,['pred_img_',num2str(id),'.npy']);

    nii_ct_gtv_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CT.nii.gz']);
    nii_cbct1_gtv_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CBCT1.nii.gz']);
    nii_cbct3_gtv_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CBCT2.nii.gz']);
    pred_gtv_File = fullfile(npy_pred_path,['pred_gtvp_',num2str(id),'.npy']);


    nii_ct_gtvn_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CT.nii.gz']);
    nii_cbct1_gtvn_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CBCT1.nii.gz']);
    nii_cbct3_gtvn_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CBCT2.nii.gz']);
    pred_gtvn_File = fullfile(npy_pred_path,['pred_gtvn_',num2str(id),'.npy']);

    %% Image
    pred_img = double(readNPY(pred_cbct3_File));
    pred_img = permute(squeeze(pred_img),[2 3 1]);
    pred_img = (pred_img+1)/2;
    ct_img = double(niftiread(nii_ct_file));
    ct_img = (ct_img+500)/1000;
    ct_img = permute(ct_img,[2,1,3]);
    cbct1_img = double(niftiread(nii_cbct1_file));
    cbct1_img = (cbct1_img+500)/1000;
    cbct1_img = permute(cbct1_img,[2,1,3]);
    cbct3_img = double(niftiread(nii_cbct3_file));
    cbct3_img = (cbct3_img+500)/1000;
    cbct3_img = permute(cbct3_img,[2,1,3]);

    % combine_img = [ct_img,cbct1_img,cbct3_img,pred_img];
    % figure,imshow3D(combine_img)
    % title('CT, Week1-CBCT, Week3-CBCT, Prediction-Week3-CBCT')
    %% body mask
    pred = readNPY(pred_file);
    % pred=squeeze(pred);
    % pred=(permute(pred,[2,3,1])+1)/2;

    ct = readNPY(fullfile(ct_path,ct_filename));
    ct = permute(ct,[2,1,3]);

    w1 = readNPY(fullfile(w1_path,cbct1_filename));
    w1 = permute(w1,[2,1,3]);

    w3 = readNPY(fullfile(w3_path,cbct3_filename));
    w3 = permute(w3,[2,1,3]);

    %% gtv
    pred_gtv = double(readNPY(pred_gtv_File));
    pred_gtv = permute(squeeze(pred_gtv),[2 3 1]);
    pred_gtv(pred_gtv>=0.5)=1;
    pred_gtv(pred_gtv<0.5)=0;
    % pred_gtv = (pred_gtv+1)/2;
    ct_gtv = double(niftiread(nii_ct_gtv_file));
    ct_gtv = ct_gtv/255;
    ct_gtv = permute(ct_gtv,[2,1,3]);
    cbct1_gtv = double(niftiread(nii_cbct1_gtv_file));
    cbct1_gtv = cbct1_gtv/255;
    cbct1_gtv = permute(cbct1_gtv,[2,1,3]);
    cbct3_gtv = double(niftiread(nii_cbct3_gtv_file));
    cbct3_gtv = cbct3_gtv/255;
    cbct3_gtv = permute(cbct3_gtv,[2,1,3]);
    
    ct_gtv_prop = [ct_gtv_prop; regionprops3(imresize3(ct_gtv,[128 128 64],'nearest'),'all')];
    w1_gtv_prop = [w1_gtv_prop; regionprops3(imresize3(cbct1_gtv,[128 128 64],'nearest'),'all')];
    w3_gtv_prop = [w3_gtv_prop; regionprops3(imresize3(cbct3_gtv,[128 128 64],'nearest'),'all')];
    pre_gtv_prop = [pre_gtv_prop; regionprops3(imresize3(pred_gtv,[128 128 64],'nearest'),'all')];
    

    %% gtvn
    pred_gtvn = double(readNPY(pred_gtvn_File));
    pred_gtvn = permute(squeeze(pred_gtvn),[2 3 1]);
    pred_gtvn(pred_gtvn>=0.5)=1;
    pred_gtvn(pred_gtvn<0.5)=0;
    % pred_gtv = (pred_gtv+1)/2;
    ct_gtvn = double(niftiread(nii_ct_gtvn_file));
    ct_gtvn = ct_gtvn/255;
    ct_gtvn = permute(ct_gtvn,[2,1,3]);
    cbct1_gtvn = double(niftiread(nii_cbct1_gtvn_file));
    cbct1_gtvn = cbct1_gtvn/255;
    cbct1_gtvn = permute(cbct1_gtvn,[2,1,3]);
    cbct3_gtvn = double(niftiread(nii_cbct3_gtvn_file));
    cbct3_gtvn = cbct3_gtvn/255;
    cbct3_gtvn = permute(cbct3_gtvn,[2,1,3]);
    
%     imresize3(ct_gtvn,[128 128 64],'nearest')
    
    ct_gtvn_prop = [ct_gtvn_prop; regionprops3(imresize3(ct_gtvn,[128 128 64],'nearest'),'all')];
    w1_gtvn_prop = [w1_gtvn_prop; regionprops3(imresize3(cbct1_gtvn,[128 128 64],'nearest'),'all')];
    w3_gtvn_prop = [w3_gtvn_prop; regionprops3(imresize3(cbct3_gtvn,[128 128 64],'nearest'),'all')];
    pre_gtvn_prop = [pre_gtvn_prop; regionprops3(imresize3(pred_gtvn,[128 128 64],'nearest'),'all')];
    %%
    % combine = [ct_img,cbct1_img,cbct3_img,pred_img; ct,w1,w3,double(pred); ct_gtv,cbct1_gtv,cbct3_gtv,pred_gtv];
    % 
    % figure,imshow3D(combine)
    % title('CT, Week1-CBCT, Week3-CBCT, Prediction-Week3-CBCT')

    dice_ct_3 = [dice_ct_3, myDice(ct,w3,1)];
    dice_1_3 = [dice_1_3, myDice(w1,w3,1)];
    dice_pred_3 = [dice_pred_3, myDice(pred,w3,1)];

    dice_gtv_ct_3 = [dice_gtv_ct_3, myDice(ct_gtv,cbct3_gtv,1)];
    dice_gtv_1_3 = [dice_gtv_1_3, myDice(cbct1_gtv,cbct3_gtv,1)];
    dice_gtv_pred_3 = [dice_gtv_pred_3, myDice(pred_gtv,cbct3_gtv,1)];

    dice_gtvn_ct_3 = [dice_gtvn_ct_3, myDice(ct_gtvn,cbct3_gtvn,1)];
    dice_gtvn_1_3 = [dice_gtvn_1_3, myDice(cbct1_gtvn,cbct3_gtvn,1)];
    dice_gtvn_pred_3 = [dice_gtvn_pred_3, myDice(pred_gtvn,cbct3_gtvn,1)];

    asd_ct_3 = [asd_ct_3, ASSDMetrics(ct,w3)];
    asd_1_3 = [asd_1_3, ASSDMetrics(w1,w3)];
    asd_pred_3 = [asd_pred_3, ASSDMetrics(pred,w3)];

    asd_gtv_ct_3 = [asd_gtv_ct_3, ASSDMetrics(ct_gtv,cbct3_gtv)];
    asd_gtv_1_3 = [asd_gtv_1_3, ASSDMetrics(cbct1_gtv,cbct3_gtv)];
    asd_gtv_pred_3 = [asd_gtv_pred_3, ASSDMetrics(pred_gtv,cbct3_gtv)];

    asd_gtvn_ct_3 = [asd_gtvn_ct_3, ASSDMetrics(ct_gtvn,cbct3_gtvn)];
    asd_gtvn_1_3 = [asd_gtvn_1_3, ASSDMetrics(cbct1_gtvn,cbct3_gtvn)];
    asd_gtvn_pred_3 = [asd_gtvn_pred_3, ASSDMetrics(pred_gtvn,cbct3_gtvn)];
    
    mse_ct_w3=[mse_ct_w3, mse(ct_img(:), cbct3_img(:))];
    mse_w1_w3=[mse_w1_w3, mse(cbct1_img(:), cbct3_img(:))];
    mse_pred_w3=[mse_pred_w3, mse(pred_img(:), cbct3_img(:))];

    
    psnr_ct_w3=[psnr_ct_w3, psnr(ct_img(:), cbct3_img(:))];
    psnr_w1_w3=[psnr_w1_w3, psnr(cbct1_img(:), cbct3_img(:))];
    psnr_pred_w3=[psnr_pred_w3, psnr(pred_img(:), cbct3_img(:))];


    ssim_ct_w3=[ssim_ct_w3, multissim3(ct_img,cbct3_img)];
    ssim_w1_w3=[ssim_w1_w3, multissim3(cbct1_img,cbct3_img)];
    ssim_pred_w3=[ssim_pred_w3, multissim3(pred_img,cbct3_img)];
    
    
    vol_ct = [vol_ct, sum(ct(:))];
    vol_w1 = [vol_w1, sum(w1(:))];
    vol_pre_cnn = [vol_pre_cnn, sum(pred(:))];
    vol_w3 = [vol_w3, sum(w3(:))];
    vol_ct_gtv = [vol_ct_gtv, sum(ct_gtv(:))];
    vol_ct_gtvn = [vol_ct_gtvn, sum(ct_gtvn(:))];
    vol_w1_gtv = [vol_w1_gtv, sum(cbct1_gtv(:))];
    vol_w1_gtvn = [vol_w1_gtvn, sum(cbct1_gtvn(:))];
    vol_pre_gtv_cnn = [vol_pre_gtv_cnn, sum(pred_gtv(:))];
    vol_pre_gtvn_cnn = [vol_pre_gtvn_cnn, sum(pred_gtvn(:))];
    vol_w3_gtv = [vol_w3_gtv, sum(cbct3_gtv(:))];
    vol_w3_gtvn = [vol_w3_gtvn, sum(cbct3_gtvn(:))];

    
%     figure,imshow3D([ct_img; cbct1_img; cbct3_img; pred_img])
%     slice_index=10;contour_slice;
%     slice_index=12;contour_slice;
%     slice_index=14;contour_slice;
%     slice_index=16;contour_slice;
%     slice_index=18;contour_slice;
%     slice_index=20;contour_slice;
%     slice_index=22;contour_slice;
%     slice_index=24;contour_slice;
%     slice_index=26;contour_slice;
    
    close all;
end

%%

[a,b]=sort(vol_w1_gtv,'descend');
figure,scatter(1:20, (vol_ct_gtv(b))/1000*16,'filled', "square") % voxel size 2*2*4 
hold on, scatter(1:20, (vol_w1_gtv(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_w3_gtv(b))/1000*16)
hold on, scatter(1:20, (vol_pre_gtv_cnn(b))/1000*16,'filled','d')
grid on
title('Testing Patients GTVp Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21','Prediction')
set(gcf, 'Position',  [100, 100, 400, 250])
% print(gcf,'Testing Patients GTVp Volume Distribution.png','-dpng','-r300'); 

figure,scatter(1:20, (vol_ct_gtv(b)-vol_w3_gtv(b))/1000*16,'filled', "square")
hold on, scatter(1:20, (vol_w1_gtv(b)-vol_w3_gtv(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_pre_gtv_cnn(b)-vol_w3_gtv(b))/1000*16,'filled','d')
grid on
title('Testing Patients GTVp Volume Difference to CBCT21')
xlabel('Patient Index')
ylabel('Volume Difference (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
% print(gcf,'Testing Patients GTVp Volume Difference Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtvn,'descend');

figure,scatter(1:20, (vol_ct_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:20, (vol_w1_gtvn(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_w3_gtvn(b))/1000*16)
hold on, scatter(1:20, (vol_pre_gtvn_cnn(b))/1000*16,'filled','d')
grid on
title('Testing Patients GTVn Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21','Prediction')
set(gcf, 'Position',  [100, 100, 400, 250])
% print(gcf,'Testing Patients GTVn Volume Distribution.png','-dpng','-r300'); 

figure,scatter(1:20, (vol_ct_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:20, (vol_w1_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_pre_gtvn_cnn(b)-vol_w3_gtvn(b))/1000*16,'filled','d')
grid on
title('Testing Patients GTVn Volume Difference to CBCT21')
xlabel('Patient Index')
ylabel('Volume Difference (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
% print(gcf,'Testing Patients GTVn Volume Difference Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtv+vol_w1_gtvn,'descend');
figure,scatter(1:20, ((vol_ct_gtv(b)+vol_ct_gtvn(b)))/1000*16,'filled', "square")
hold on, scatter(1:20, ((vol_w1_gtv(b)+vol_w1_gtvn(b)))/1000*16,'filled')
hold on, scatter(1:20, (vol_w3_gtv(b)+vol_w3_gtvn(b))/1000*16)
hold on, scatter(1:20, ((vol_pre_gtv_cnn(b)+vol_pre_gtvn_cnn(b)))/1000*16,'filled','d')
grid on
title('Testing Data GTV Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21','Prediction')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,20])
print(gcf,'Testing Data GTV Volume Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtv+vol_w1_gtvn,'descend');
figure,scatter(1:20, ((vol_ct_gtv(b)+vol_ct_gtvn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled', "square")
hold on, scatter(1:20, ((vol_w1_gtv(b)+vol_w1_gtvn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled')
hold on, scatter(1:20, ((vol_pre_gtv_cnn(b)+vol_pre_gtvn_cnn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled','d')
grid on
title('Testing Data GTV Volume Difference Distribution')
xlabel('Patient Index')
ylabel('Volume Difference (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,20])
% print(gcf,'Testing Data GTV Volume Change Distribution.png','-dpng','-r300'); 

%% functions
function dice = myDice(volume1, volume2, mu)
    overlap = volume1.*volume2;
    overlap = sum(overlap(:));
    
    dice = (2*overlap+mu)/(sum(volume1(:)+volume2(:))+mu);
end