%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion 
%REference taken form https://www.mathworks.com/help/stats/confusionmat.html

pred_matrix = zeros(1,size(xtest,2));

 for i=1:100:size(xtest, 2)
     [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
     [value, index] = max(P, [], 1);
     pred_matrix(:, i:i+99) = index;
 end
Confusion_matrix = confusionmat(ytest, pred_matrix);
confusionchart(Confusion_matrix, [0:9])