function h = sigmoid(z);
    h = 1 ./ (1 + exp(-z));
end 