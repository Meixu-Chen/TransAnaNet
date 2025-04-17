
% figure,imshow3D([ct_img; cbct1_img; cbct3_img; pred_img])

figure,
subplot(2,4,1)
imshow(ct_img(:,:, slice_index)), hold on
contour(ct_gtv(:,:,slice_index),'r'), contour(ct_gtvn(:,:,slice_index),'b'), hold off
title('CT')

subplot(2,4,2)
imshow(cbct1_img(:,:, slice_index)), hold on
contour(cbct1_gtv(:,:,slice_index),'r'), contour(cbct1_gtvn(:,:,slice_index),'b'), hold off
title('CBCT01')

subplot(2,4,3)
imshow(cbct3_img(:,:, slice_index)), hold on
contour(cbct3_gtv(:,:,slice_index),'r'), contour(cbct3_gtvn(:,:,slice_index),'b'), hold off
title('CBCT21')

subplot(2,4,4)
imshow(pred_img(:,:, slice_index)), hold on
contour(pred_gtv(:,:,slice_index),'r'), contour(pred_gtvn(:,:,slice_index),'b'), hold off
title('Prediction')

% figure,
subplot(2,4,5)
temp1 = ct_img(:,:, slice_index)-cbct3_img(:,:, slice_index);
% temp2 = ct_gtv(:,:, slice_index)-cbct3_gtv(:,:, slice_index);
% temp3 = ct_gtvn(:,:, slice_index)-cbct3_gtvn(:,:, slice_index);
imshow(temp1, [-0.2,0.2])
title('CT')

subplot(2,4,6)
temp1 = cbct1_img(:,:, slice_index)-cbct3_img(:,:, slice_index);
% temp2 = cbct1_gtv(:,:, slice_index)-cbct3_gtv(:,:, slice_index);
% temp3 = cbct1_gtvn(:,:, slice_index)-cbct3_gtvn(:,:, slice_index);
imshow(temp1, [-0.2,0.2])
title('CBCT01')

subplot(2,4,7)
imshow(cbct3_img(:,:, slice_index)-cbct3_img(:,:, slice_index), [-0.2,0.2])
title('CBCT21')

subplot(2,4,8)
temp1 = pred_img(:,:, slice_index)-cbct3_img(:,:, slice_index);
% temp2 = pred_gtv(:,:, slice_index)-cbct3_gtv(:,:, slice_index);
% temp3 = pred_gtvn(:,:, slice_index)-cbct3_gtvn(:,:, slice_index);
imshow(temp1, [-0.2,0.2])
title('Prediction')

% % figure,
% subplot(2,4,5)
% imshow(ct(:,:, slice_index)-w3(:,:, slice_index), [])
% title('CT')
% 
% subplot(2,4,6)
% imshow(w1(:,:, slice_index)-w3(:,:, slice_index), [])
% title('CBCT01')
% 
% subplot(2,4,7)
% imshow(w3(:,:, slice_index)-w3(:,:, slice_index)+0.5, [0,1])
% title('CBCT21')
% 
% subplot(2,4,8)
% imshow(pred(:,:, slice_index)-w3(:,:, slice_index), [])
% title('Prediction')

