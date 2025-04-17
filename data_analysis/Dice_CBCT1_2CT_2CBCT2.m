label_path = './label';
input_path = './input';

CBCTs = dir(fullfile(label_path, '*_GTVn_CBCT2.nii.gz'));
dice = zeros(length(CBCTs),4);
for i = 1:length(CBCTs)
    patient = CBCTs(i).name;
    patient = patient(1:6);
    
    CT_gtvp_path = fullfile(input_path, [patient, '_GTVp_CT.nii.gz']);
    CT_gtvn_path = fullfile(input_path, [patient, '_GTVn_CT.nii.gz']);
    CBCT1_gtvp_path = fullfile(input_path, [patient, '_GTVp_CBCT1.nii.gz']);
    CBCT1_gtvn_path = fullfile(input_path, [patient, '_GTVn_CBCT1.nii.gz']);
    CBCT2_gtvp_path = fullfile(label_path, [patient, '_GTVp_CBCT2.nii.gz']);
    CBCT2_gtvn_path = fullfile(label_path, [patient, '_GTVn_CBCT2.nii.gz']);
    
    CT_gtvp = double(niftiread(CT_gtvp_path)>128);
    CT_gtvn = double(niftiread(CT_gtvn_path)>128);
    CBCT2_gtvp = double(niftiread(CBCT2_gtvp_path)>128);
    CBCT2_gtvn = double(niftiread(CBCT2_gtvn_path)>128);
    CBCT1_gtvp = double(niftiread(CBCT1_gtvp_path)>128);
    CBCT1_gtvn = double(niftiread(CBCT1_gtvn_path)>128);
    
    dice(i,1) = myDice(CT_gtvp, CBCT2_gtvp, 1);
    dice(i,2) = myDice(CBCT1_gtvp, CBCT2_gtvp, 1);
    dice(i,3) = myDice(CT_gtvn, CBCT2_gtvn, 1);
    dice(i,4) = myDice(CBCT1_gtvn, CBCT2_gtvn, 1);
end

%% functions
function dice = myDice(volume1, volume2, mu)
    overlap = volume1.*volume2;
    overlap = sum(overlap(:));
    
    dice = (2*overlap+mu)/(sum(volume1(:)+volume2(:))+mu);
end