function [J, grad] = computeCost(theta,x,y,lambda)
    [m,n] = size(x);
    x = [ones(m,1), x];
    % hø = g(x * ø)
    h = sigmoid(x * theta);
    
    % excluded the first theta value
    theta1 = [0 ; theta(2:size(theta), :)];

    % penalize the thetas 
    p = lambda*(theta1'*theta1)/(2*m);
    J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;

    % calculate grads
    grad = (x'*(h - y)+lambda*theta1)/m;

end

