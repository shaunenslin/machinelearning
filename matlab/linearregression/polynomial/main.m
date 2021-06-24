clc;
clear;
close all;

% declare our quadratic polynomial
f = @(x) 2*x.^2 - 4*x + 5;

% Create our x axis by getting some random uniform x vales between -1 and 5
x = unifrnd(-1, 5, [100,1]);

% Calc the y values for each x, by running the polynomial but add some variations
y = f(x) + 1.5 * randn(size(x));

% plot the data points
figure;
plot(x,y,'rx');

% find the best coeffeciant
[p,S] = polyfit(x,y,2);

% Create a new xx to plot coeffecient
xx = linspace(-1, 5, 100);

% Evaluate our polynomial
[yy, delta] = polyval(p,xx,S);
hold on;
plot(xx,yy);

% Lets use our delta to plot our support and resistance lines
plot(xx, yy + 2*delta, 'b:');
plot(xx, yy - 2*delta, 'b:');


