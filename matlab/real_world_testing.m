%% Network defintion
layers = get_lenet();
layers{1}.batch_size = 1; 

%load the train weights
load lenet.mat

% Reading "img_num" of test images from the relative path 
for img_num = 1:5
    path = sprintf('../images/real_testing/testing_image%d.jpg', img_num);
    test_image = imread(path);
    
    %changing the dimesnions of image from RGB(3) to gray scale 
    if ndims(test_image)==3
        test_image = rgb2gray(test_image);
    end
     
    test_image = imresize(test_image, [28,28]);
    test_image = double(test_image)/255;
    test_image = reshape(test_image,[],1);

    [~, P] = convnet_forward(params, layers, test_image(:));
    [val, index] = max(P);

    fprintf('Prediction for this test-image is: %d.\n', index-1);
    
end
