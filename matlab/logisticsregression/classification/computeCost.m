function [J,grad] = computeCost(theta, X, y, lambda) 
    % hypothesis
    h = sigmoid(X*theta);
    m = size(X,1);
    
    % new theta for lambda, excluding col 1
    theta1 = [0; theta(2:size(theta),:)];
    % penalise thetas to reduce cost
    p = lambda*(theta1'*theta)/(2*m);
    J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;
    grad = (X'*(h - y) + lambda * theta1)/m;
end