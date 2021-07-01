clear;
% open csv file
tbl = readtable('test.csv');

% replace strings with labels
ds(:,1) = grp2idx(tbl{:,2});
ds(:,2) = grp2idx(tbl{:,3});
ds(:,3) = tbl{:,4};
ds(:,4) = grp2idx(tbl{:,5});
ds(:,5) = grp2idx(tbl{:,6});
ds(:,6) = tbl{:,7};
ds(:,7) = grp2idx(tbl{:,8});
ds(:,8) = tbl{:,9};
[ds(:,9),labels] = grp2idx(tbl{:,10});

% remove NaN
ds = rmmissing(ds);
[m,n] = size(ds);

X = [ones(m,1) ds(:,[2 4 8])]; % [2 4 8]
y = ds(:,n);

% setups
[m,n] = size(X);
lambda = 0.01;
thetas = zeros(length(labels),n);

% loop through labels and run cost optimisations
for i = 1:length(labels)
    itheta = zeros(n,1);
    options = optimset('GradObj', 'On', 'MaxIter', 400);
    theta = fminunc(@(t)computeCost(t, X, (y == i), lambda), itheta, options);
    thetas(i,:) = theta';
end

% predict results
p = predict(thetas,X);
accuracy = mean((p == y)* 100)

