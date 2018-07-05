clear all;
load('mnist.mat')

mu = mean(mean(X_train));

X_train = X_train - mu;
X_test = X_test - mu;

acc = train_and_test( X_train, Y_train, X_test, Y_test, mu );
fprintf('The accuracy of the implementation is: %f\n', acc);
