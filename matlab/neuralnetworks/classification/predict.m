function p = predict(Theta1, Theta2, X)
    %PREDICT Predict the label of an input given a trained neural network
    m = size(X, 1);

    h1 = sigmoid([ones(m, 1) X] * Theta1');
    h2 = sigmoid([ones(m, 1) h1] * Theta2');
    [dummy, p] = max(h2, [], 2);

end
