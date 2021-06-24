
clc;clear;
% load data
T = readtable('heart.csv');
ds = T{:,:};
x = ds(:,1:size(ds,2)-1);
y = ds(:,size(ds,2));

% compute cost and gradient
[m,n] = size(x);
theta = zeros(n+1,1);
lambda = 7;
[J, grad] = computeCost(theta,x,y,lambda);

% SpecifyObjectiveGradient
options = optimset('GradObj','On','MaxIter',400);
% Run cost optimisation with "Find minimum of unconstrained multivariable" function
theta = fminunc(@(t)computeCost(t, x, y, lambda), theta, options)

% check predictions by predicting all our x's and lets see if 
p = predict(theta, x);
accuracy = mean((p == y) * 100)

% predict a result close to row 10
f = [52,1,2,170,201,1,1,160,0,0.6,2,0,3];
yes = predict(theta, f)

% predict a result close to row 200
f = [61,1,0,118,260,0,1,94,1,1.6,1,2,3];
no = predict(theta, f)

