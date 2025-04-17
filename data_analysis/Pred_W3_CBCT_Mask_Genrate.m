folder = 'pred_val_better';
savefolder = 'Pred_W3_better';
mkdir(savefolder)
patients = dir([folder,'/pred_img_*.npy']);
fprintf('there are %d patients data\n',length(patients));

threshold =0.35;

for i = 1:length(patients)
    fprintf('processing %dth patient %s\n',i,patients(i).name);
    data = readNPY([patients(i).folder,'/',patients(i).name]);
    data = permute(squeeze(data),[2 3 1]);
    data = (data+1)/2;
    [x,y,z]=size(data);
    if (x~=128)||(y~=128)||(z~=32)
        warning('somthing wrong')
    end
    
    mask = zeros(size(data));
    for s = 1:z
        binary_image = data(:,:,s)>threshold;
        stats = regionprops(binary_image,'Area','PixelIdxList');
        mask_slice = zeros(size(binary_image));
        if length(stats)>1
            max_index = 1;
            for regin =1:length(stats)
                if stats(regin).Area>stats(max_index).Area
                    max_index = regin;
                end
            end
            index = stats(max_index).PixelIdxList;
            mask_slice(index)=1;
        elseif length(stats)<1
            continue;
        else
            index = stats.PixelIdxList;
            mask_slice(index)=1;
        end
        se = strel('sphere',5);
        mask_slice = imdilate(mask_slice,se);
        mask_slice = imerode(mask_slice,se);
        mask(:,:,s)=imfill(mask_slice,'holes');
    end
    mask = imdilate(mask,se);
    mask = imerode(mask,se);
    mask=imfill(mask,'holes');
    
    savename = split(patients(i).name,'.');
    savename = fullfile(savefolder,[savename{1},'.npy']);
    writeNPY(mask, savename);
end