
% This script is ready to visualize the result of the image1.
% The image is read and stored in the variable input_image1 as shown below.
%For geting visuals of other input images, please refer the comments
%Some portion of the code is same for input image and is mentioned in the
%comments
% Different portions are either provided or asked to change the varibale
% according to the input image as mentioned in the comments.

%Visual resuls are provided in the report for all 4 input images. 

layers = get_lenet();
load lenet.mat
layers{1}.batch_size = 1;

%Read the images from their paths
input_image1 = imread('../images/image1.jpg');
input_image2 = imread('../images/image2.jpg');
input_image3 = imread('../images/image3.png');
input_image4 = imread('../images/image4.jpg');


%For input image1
% For other input images please change the input_image1 to input_image{2,3,4}
input_image1 = rgb2gray(input_image1);
input_image1 = imbinarize(input_image1);
final_input_image1 = im2double(input_image1);

%Using the bwlabel on input image
%For other image, change the variable input_image1 to input_image{2,3,4} 
[L,n] = bwlabel(1.-input_image1); 
bb = cell(1,n);


 %finding connected components 
 % where arrX and arrY are row and column of matrix respectively
 %Note: Below portion remain same for every input images. 
for count = 1:n
    [arrX,arrY] = find(L==count)
    bb{count} = input_image1(min(arrX):max(arrX), min(arrY):max(arrY)); 
    min_of_arrX = min(arrX);
    min_of_arrY = min(arrY);
    [arrX,arrY] = size(bb{count});

    %calling the local built get_padding function which does padding 
    %This call is applicable for every input image. 
    XY_reshape = get_padding(arrX, arrY, count, bb);

%Note: Below code is aplicable and must be included for every input image. 
% REference for the following is https://www.mathworks.com/help/matlab/ref/imresize.html
    bb{count} = imresize(bb{count}, [28-2*XY_reshape 28-2*XY_reshape], 'nearest')';
    bb{count} = padarray(bb{count}, [XY_reshape XY_reshape], 1);

    [~, P] = convnet_forward(params, layers, logical(1.-bb{count}(:)));
    [~, index] = max(P); 
   
    %subploting and insertText Function to store final_input_image1 array

    %% Use the below for input_image1
    bb{count} = transpose(bb{count});
    bb{count} = im2double(bb{count});
    subplot(2,5,count);
    imshow(bb{count});

    % https://www.mathworks.com/help/vision/ref/inserttext.html
    %https://www.mathworks.com/matlabcentral/answers/1605080-filling-a-region-of-a-grayscale-image-with-a-colour-corresponding-to-colorbar
    final_input_image1 = insertText(final_input_image1, [min_of_arrY min_of_arrX], index-1, 'FontSize', 64); 

 %% Use the below for input_image2
%     bb{count} = transpose(bb{count});
%     bb{count} = im2double(bb{count});
%     subplot(2,5,count);
%     imshow(bb{count});
% 
%     final_input_image2 = insertText(final_input_image2, [min_of_arrY min_of_arrX], index-1, 'FontSize', 64); 

  %% Use the below for input_image3
%     bb{count} = transpose(bb{count});
%     bb{count} = im2double(bb{count});
%     subplot(1,6,count);
%     imshow(bb{count});
% 
%     final_input_image3 = insertText(final_input_image3, [min_of_arrY min_of_arrX], index-1, 'FontSize', 32);

%% Use the below for input_image4
% bb{count} = transpose(bb{count});
%     bb{count} = im2double(bb{count});
%     subplot(3,17,count);
%     imshow(bb{count});
% 
%     final_input_image4 = insertText(final_input_image4, [min_of_arrY min_of_arrX], index-1, 'FontSize', 16); 
       
end

% Image show function calling
% Please change the iinput_image1 variable to input_image{2,3,4}. 
figure; imshow(final_input_image1);


% get padding local function
function f = get_padding(arrX, arrY, count, bb)
    if arrX > arrY
        bb{count} = padarray(bb{count}, [0 floor((arrX-arrY)/2)], 1);
    else
        bb{count} = padarray(bb{count}, [floor((arrY-arrX)/2) 0], 1); 
    end
    
    if max([arrX, arrY]) > 100 && (arrX/arrY-1) > 0.5
        f = 5;
    else
        if max([arrX, arrY]) > 100
            f = 4;
        else
            f = 0;
        end
    end
   
end



