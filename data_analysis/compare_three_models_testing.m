load('test_TransAna_vol.mat')
load('test_vit_vol.mat')
load('test_cnn_vol.mat')

[a,b]=sort(vol_w1_gtv,'descend');

figure,scatter(1:20, (vol_ct_gtv(b)-vol_w3_gtv(b))/1000*16,'filled', "square")
hold on, scatter(1:20, (vol_w1_gtv(b)-vol_w3_gtv(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_pre_gtv(b)-vol_w3_gtv(b))/1000*16,'filled','d')
hold on, scatter(1:20, (vol_pre_gtv_vit(b)-vol_w3_gtv(b))/1000*16,'filled',"^")
hold on, scatter(1:20, (vol_pre_gtv_cnn(b)-vol_w3_gtv(b))/1000*16,'filled',">")
grid on
ylim([0,35])
title('Testing Patient GTVp Volume Prediction Error')
xlabel('Patient Index')
ylabel('Volume Prediction Error (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','TransAnaNet Prediction to CBCT21','ViT Prediction to CBCT21','CNN Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
print(gcf,'Testing Patients GTVp Volume Difference Distribution V2.png','-dpng','-r300'); 

[a,b]=sort(vol_w1_gtvn,'descend');

% figure,scatter(1:20, (vol_ct_gtvn(b))/1000*16,'filled', "square")
% hold on, scatter(1:20, (vol_w1_gtvn(b))/1000*16,'filled')
% hold on, scatter(1:20, (vol_pre_gtvn(b))/1000*16,'filled','d')
% hold on, scatter(1:20, (vol_pre_gtvn_vit(b))/1000*16,'filled',"^")
% hold on, scatter(1:20, (vol_pre_gtvn_cnn(b))/1000*16,'filled',">")
% hold on, scatter(1:20, (vol_w3_gtvn(b))/1000*16,'filled', "hexagram","k")
% grid on
% title('Testing Patients GTVn Volume Distribution')
% xlabel('Patient Index')
% ylabel('Volume (cc)')
% legend('CT','CBCT01','TransAnaNet Prediction','ViT Prediction','CNN Prediction','CBCT21')
% set(gcf, 'Position',  [100, 100, 400, 250])
% print(gcf,'Testing Patients GTVn Volume Distribution V2.png','-dpng','-r300'); 

figure,scatter(1:20, (vol_ct_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled', "square")
hold on, scatter(1:20, (vol_w1_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled')
hold on, scatter(1:20, (vol_pre_gtvn(b)-vol_w3_gtvn(b))/1000*16,'filled','d')
hold on, scatter(1:20, (vol_pre_gtvn_vit(b)-vol_w3_gtvn(b))/1000*16,'filled',"^")
hold on, scatter(1:20, (vol_pre_gtvn_cnn(b)-vol_w3_gtvn(b))/1000*16,'filled',">")
grid on
ylim([0,35])
title('Testing Patient GTVn Volume Prediction Error')
xlabel('Patient Index')
ylabel('Volume Prediction Error (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','TransAnaNet Prediction to CBCT21','ViT Prediction to CBCT21','CNN Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
print(gcf,'Testing Patients GTVn Volume Difference Distribution V2.png','-dpng','-r300'); 


[a,b]=sort(vol_w1_gtv+vol_w1_gtvn,'descend');
figure,scatter(1:20, ((vol_ct_gtv(b)+vol_ct_gtvn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled', "square")
hold on, scatter(1:20, ((vol_w1_gtv(b)+vol_w1_gtvn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled')
hold on, scatter(1:20, ((vol_pre_gtv(b)+vol_pre_gtvn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled','d')
hold on, scatter(1:20, ((vol_pre_gtv_vit(b)+vol_pre_gtvn_vit(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled',"^")
hold on, scatter(1:20, ((vol_pre_gtv_cnn(b)+vol_pre_gtvn_cnn(b))-(vol_w3_gtv(b)+vol_w3_gtvn(b)))/1000*16,'filled',">")
grid on
ylim([0,35])
title('Testing Patient GTV Volume Prediction Error')
xlabel('Patient Index')
ylabel('Volume Prediction Error (cc)')
legend('CT to CBCT21','CBCT01 to CBCT21','TransAnaNet Prediction to CBCT21','ViT Prediction to CBCT21','CNN Prediction to CBCT21')
set(gcf, 'Position',  [100, 100, 400, 250])
xlim([0,20])
print(gcf,'Testing Data GTV Volume Change Distribution V2.png','-dpng','-r300'); 