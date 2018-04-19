clear all;
clc;
load MNIST_digit_data.mat
whos
% [labels_train, images_train] = libsvmread('MNIST_digit_data');
%%% randomly permute data points
rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);

% %%% if you want to use only the first 1000 data points.
 images_train = images_train(1:1000, :);
 labels_train = labels_train(1:1000, :);
 
 
 model = svmtrain(labels_train, images_train, '-t 0');
 [predicted_label, accuracy, decision_values] = svmpredict(labels_test, images_test, model);
 correct = 0;
 for i = 1: 1000
    if predicted_label(i,1) == labels_test(i,1)
        correct = correct + 1;
    end
 end
 
 
 my_accuracy = (correct/1000)*100;
fprintf('Accuracy for 1000 iterations is %2.4f\n',my_accuracy);

 
 
 
 