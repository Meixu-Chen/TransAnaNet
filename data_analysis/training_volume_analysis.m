% clear,clc,close all

ct_path= '../CT_mask';
w1_path = '../W1_CBCT_mask';
w3_path = '../W3_CBCT_mask';
nii_input_path= '../../input';
nii_label_path = '../../label';

index = readtable('data_loader_training.txt');
index = table2cell(index);

vol_ct = [];
vol_w1 = [];
vol_w3 = [];
vol_ct_gtv = [];
vol_ct_gtvn = [];
vol_w1_gtv = [];
vol_w1_gtvn = [];
% vol_pre_gtv = [];
% vol_pre_gtvn = [];
vol_w3_gtv = [];
vol_w3_gtvn = [];

% regionprop for measuring center of mass difference and max-diameter
ct_gtv_prop = [];
ct_gtvn_prop = [];
w1_gtv_prop = [];
w1_gtvn_prop = [];
w3_gtv_prop = [];
w3_gtvn_prop = [];

for id = 0:length(index)-1
    
    ct_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CT.npy'];

    % files = dir(fullfile(w1_path,'*.npy'));
    % cbct1_filename = files(id+1).name;
    cbct1_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CBCT1.npy'];

    % files = dir(fullfile(w3_path,'*.npy'));
    % cbct3_filename = files(id+1).name;
    cbct3_filename = ['PT_', num2str(index{id+1,7},'%03d'),'_CBCT2.npy'];


    nii_ct_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CT.nii.gz']);
    nii_cbct1_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CBCT1.nii.gz']);
    nii_cbct3_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_CBCT2.nii.gz']);

    nii_ct_gtv_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CT.nii.gz']);
    nii_cbct1_gtv_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CBCT1.nii.gz']);
    nii_cbct3_gtv_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVp_CBCT2.nii.gz']);


    nii_ct_gtvn_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CT.nii.gz']);
    nii_cbct1_gtvn_file = fullfile(nii_input_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CBCT1.nii.gz']);
    nii_cbct3_gtvn_file = fullfile(nii_label_path, ['Pt_', num2str(index{id+1,7},'%03d'),'_GTVn_CBCT2.nii.gz']);


    ct_img = double(niftiread(nii_ct_file));
    ct_img = (ct_img+500)/1000;
    ct_img = permute(ct_img,[2,1,3]);
    cbct1_img = double(niftiread(nii_cbct1_file));
    cbct1_img = (cbct1_img+500)/1000;
    cbct1_img = permute(cbct1_img,[2,1,3]);
    cbct3_img = double(niftiread(nii_cbct3_file));
    cbct3_img = (cbct3_img+500)/1000;
    cbct3_img = permute(cbct3_img,[2,1,3]);

    ct = readNPY(fullfile(ct_path,ct_filename));
    ct = permute(ct,[2,1,3]);

    w1 = readNPY(fullfile(w1_path,cbct1_filename));
    w1 = permute(w1,[2,1,3]);

    w3 = readNPY(fullfile(w3_path,cbct3_filename));
    w3 = permute(w3,[2,1,3]);

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

    ct_gtvn = double(niftiread(nii_ct_gtvn_file));
    ct_gtvn = ct_gtvn/255;
    ct_gtvn = permute(ct_gtvn,[2,1,3]);
    cbct1_gtvn = double(niftiread(nii_cbct1_gtvn_file));
    cbct1_gtvn = cbct1_gtvn/255;
    cbct1_gtvn = permute(cbct1_gtvn,[2,1,3]);
    cbct3_gtvn = double(niftiread(nii_cbct3_gtvn_file));
    cbct3_gtvn = cbct3_gtvn/255;
    cbct3_gtvn = permute(cbct3_gtvn,[2,1,3]);
    
    ct_gtvn_prop = [ct_gtvn_prop; regionprops3(imresize3(ct_gtvn,[128 128 64],'nearest'),'all')];
    w1_gtvn_prop = [w1_gtvn_prop; regionprops3(imresize3(cbct1_gtvn,[128 128 64],'nearest'),'all')];
    w3_gtvn_prop = [w3_gtvn_prop; regionprops3(imresize3(cbct3_gtvn,[128 128 64],'nearest'),'all')];
    
    vol_ct = [vol_ct, sum(ct(:))];
    vol_w1 = [vol_w1, sum(w1(:))];
    vol_w3 = [vol_w3, sum(w3(:))];
    vol_ct_gtv = [vol_ct_gtv, sum(ct_gtv(:))];
    vol_ct_gtvn = [vol_ct_gtvn, sum(ct_gtvn(:))];
    vol_w1_gtv = [vol_w1_gtv, sum(cbct1_gtv(:))];
    vol_w1_gtvn = [vol_w1_gtvn, sum(cbct1_gtvn(:))];
    vol_w3_gtv = [vol_w3_gtv, sum(cbct3_gtv(:))];
    vol_w3_gtvn = [vol_w3_gtvn, sum(cbct3_gtvn(:))];


end

%%

[a,b]=sort(vol_w1_gtv,'descend');
figure,scatter(1:101, (vol_ct_gtv(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b))/1000*16,'filled')
hold on, scatter(1:101, (vol_w3_gtv(b))/1000*16)
grid on
title('Training Data GTVp Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])

print(gcf,'Training Data GTVp Volume Distribution.png','-dpng','-r300'); 

figure,scatter(1:101, (vol_ct_gtv(b)-vol_w3_gtv(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b)-vol_w3_gtv(b))/1000*16,'filled')
grid on
title('Training Data GTVp Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTVp Volume Change Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtvn,'descend');
figure,scatter(1:101, (vol_ct_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtvn(b))/1000*16,'filled')
hold on, scatter(1:101, (vol_w3_gtvn(b))/1000*16)
grid on
title('Training Data GTVn Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTVn Volume Distribution.png','-dpng','-r300'); 

figure,scatter(1:101, (vol_ct_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled')
grid on
title('Training Data GTVn Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTVn Volume Change Distribution.png','-dpng','-r300'); 


[a,b]=sort(vol_w1_gtv+vol_w1_gtvn,'descend');
figure,scatter(1:101, (vol_ct_gtv(b)+vol_ct_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b)+vol_w1_gtvn(b))/1000*16,'filled')
hold on, scatter(1:101, (vol_w3_gtv(b)+vol_w3_gtvn(b))/1000*16)
grid on
title('Training Data GTV Volume Distribution')
xlabel('Patient Index')
ylabel('Volume (cc)')
legend('CT','CBCT01','CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])

print(gcf,'Training Data GTV Volume Distribution.png','-dpng','-r300'); 

figure,scatter(1:101, (vol_ct_gtv(b)-vol_w3_gtv(b)+vol_ct_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b)-vol_w3_gtv(b)+vol_w1_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled')
grid on
title('Training Data GTV Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTV Volume Change Distribution.png','-dpng','-r300'); 

%%
[a,b]=sort(vol_w1_gtv,'descend');
figure,scatter(1:101, (vol_ct_gtv(b)./vol_w3_gtv(b)-1)*100,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b)./vol_w3_gtv(b)-1)*100,'filled')
grid on
title('Training Data GTVp Relative Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease ratio')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
ytickformat('percentage')
print(gcf,'Training Data GTVp Relative Volume Change Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtvn,'descend');
figure,scatter(1:101, (vol_ct_gtvn(b)./vol_w3_gtvn(b)-1)*100,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtvn(b)./vol_w3_gtvn(b)-1)*100,'filled')
grid on
title('Training Data GTVn Relative Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease ratio')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
ytickformat('percentage')
print(gcf,'Training Data GTVn Relative Volume Change Distribution.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtv+vol_w1_gtvn,'descend');
figure,scatter(1:101, (vol_ct_gtv(b)+vol_ct_gtvn(b))./(vol_w3_gtv(b)+vol_w3_gtvn(b))*100-100,'filled', "square")
hold on, scatter(1:101, (vol_w1_gtv(b)+vol_w1_gtvn(b))./(vol_w3_gtv(b)+vol_w3_gtvn(b))*100-100,'filled')
grid on
title('Training Data GTV Relative Volume Change Distribution')
xlabel('Patient Index')
ylabel('Volume Decrease ratio')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
ytickformat('percentage')
print(gcf,'Training Data GTV Relative Volume Change Distribution.png','-dpng','-r300'); 

%%
figure, boxplot((vol_w1_gtv)./(vol_w3_gtv)*100-100,'Labels',{'CBCT01 to CBCT21'})
ylabel('Relative Volume Change')
title('GTVp Relative Volume Change Distribution')
ytickformat('percentage')
set(gcf, 'Position',  [100, 100, 400, 250])
grid on
print(gcf,'Training GTVp Relative Volume Change Boxplot.png','-dpng','-r300'); 

figure, boxplot((vol_w1_gtvn)./(vol_w3_gtvn)*100-100,'Labels',{'CBCT01 to CBCT21'})
ylabel('Relative Volume Change')
title('GTVn Relative Volume Change Distribution')
ytickformat('percentage')
set(gcf, 'Position',  [100, 100, 400, 250])
grid on
print(gcf,'Training GTVn Relative Volume Change Boxplot.png','-dpng','-r300'); 

figure, boxplot((vol_w1_gtv(b)+vol_w1_gtvn(b))./(vol_w3_gtv(b)+vol_w3_gtvn(b))*100-100,'Labels',{'CBCT01 to CBCT21'})
ylabel('Relative Volume Change')
title('GTV Relative Volume Change Distribution')
ytickformat('percentage')
set(gcf, 'Position',  [100, 100, 400, 250])
grid on
print(gcf,'Training GTV Relative Volume Change Boxplot.png','-dpng','-r300'); 
%%
[a,b]=sort(vol_w1_gtv,'descend');
dis_ct_w3_gtv = zeros(1,length(b));
dis_w1_w3_gtv = zeros(1,length(b));
for i =1:length(b)
    dis_ct_w3_gtv(i) = norm(ct_gtv_prop.Centroid(b(i),:)-w3_gtv_prop.Centroid(b(i),:))*2; 
    dis_w1_w3_gtv(i) = norm(w1_gtv_prop.Centroid(b(i),:)-w3_gtv_prop.Centroid(b(i),:))*2; 
end
figure,boxplot(dis_w1_w3_gtv)

dis_ct_w3_gtv(dis_ct_w3_gtv>40)=dis_ct_w3_gtv(dis_ct_w3_gtv>40)-40;
dis_w1_w3_gtv(dis_w1_w3_gtv>40)=dis_w1_w3_gtv(dis_w1_w3_gtv>40)-40;

figure,scatter(1:101, dis_ct_w3_gtv(b),'filled', "square")
hold on, scatter(1:101, dis_w1_w3_gtv(b),'filled')
grid on
title('Training Data GTVp Center of Mass Change Distribution')
xlabel('Patient Index')
ylabel('Distance (mm)')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTVp Center of Mass Change Distribution.png','-dpng','-r300'); 


dis_ct_w3_gtvn = zeros(1,length(vol_w1_gtvn));
dis_w1_w3_gtvn = zeros(1,length(vol_w1_gtvn));
count = 0;
for i =1:length(vol_w1_gtvn)-1
    if vol_w1_gtvn(i)==0
        count = count+1;
        continue
    else
        dis_ct_w3_gtvn(i) = norm(ct_gtvn_prop.Centroid(i-count,:)-w3_gtvn_prop.Centroid(i-count,:))*2; 
        dis_w1_w3_gtvn(i) = norm(w1_gtvn_prop.Centroid(i-count,:)-w3_gtvn_prop.Centroid(i-count,:))*2; 
    end
end
dis_ct_w3_gtvn(dis_ct_w3_gtvn>40)=dis_ct_w3_gtvn(dis_ct_w3_gtvn>40)-40;
dis_w1_w3_gtvn(dis_w1_w3_gtvn>40)=dis_w1_w3_gtvn(dis_w1_w3_gtvn>40)-40;
% figure,boxplot(dis)
figure,boxplot(dis_w1_w3_gtvn)

[a,b]=sort(vol_w1_gtvn,'descend');
figure,scatter(1:101, dis_ct_w3_gtvn(b),'filled', "square")
hold on, scatter(1:101, dis_w1_w3_gtvn(b),'filled')
grid on
title('Training Data GTVn Center of Mass Change Distribution')
xlabel('Patient Index')
ylabel('Distance (mm)')
legend('CT to CBCT21','CBCT01 to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,101])
print(gcf,'Training Data GTVn Center of Mass Change Distribution.png','-dpng','-r300'); 

figure, boxplot(dis_w1_w3_gtv,'Labels',{'CBCT01 to CBCT21'})
ylabel('Center of Mass Shift (mm)')
title('GTVp Center of Mass Shift Distribution')
set(gcf, 'Position',  [100, 100, 400, 250])
grid on
print(gcf,'Training GTVp Center of Mass Shift.png','-dpng','-r300'); 

num = length(find(vol_w1_gtvn>0));
figure, boxplot(dis_w1_w3_gtvn(vol_w1_gtvn>0),'Labels',{'CBCT01 to CBCT21'})
ylabel('Center of Mass Shift (mm)')
title('GTVn Center of Mass Shift Distribution')
set(gcf, 'Position',  [100, 100, 400, 250])
grid on
print(gcf,'Training GTVn Center of Mass Shift.png','-dpng','-r300'); 