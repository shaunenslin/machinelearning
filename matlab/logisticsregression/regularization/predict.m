function p = predict(theta, x)
    x = [ones(size(x,1),1),x];
    % get hypothesis for all x values, but round to zero or 1
    p = round(sigmoid(x * theta));
end