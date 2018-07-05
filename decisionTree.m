
clear all;

%Read training set and test set
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

%Size of test set
test_scale = size(test_set)

%Train model and predict labels
model = fitctree(train_set,train_label);

%Predict labels on test data
pred_label = predict(model,test_set);

%10-fold Cross Validation
cvmodel = crossval(model);
cvmdlloss = kfoldLoss(cvmodel);

%Accuracy
num_correct = sum(test_label==pred_label);
accuracy = num_correct / test_scale(1);
disp(accuracy)

%Confusion Matrix
C = confusionmat(test_label,pred_label)
plotconfusion(test_label,pred_label)

