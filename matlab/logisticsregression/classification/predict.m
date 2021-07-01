function p = predict(thetas, X)
    predict = sigmoid(X*thetas');
    [predictmax,p] = max(predict,[],2);
end