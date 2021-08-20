from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv('customertrain.csv')
df.head()

# Drop rows with no Output values
df.dropna(subset=['Var_1'], inplace=True)

# extract Y and drop from dataframe
Y = df["Var_1"]
df = df.drop(["Var_1"], axis=1)

# encode class values as integers
yencoder = LabelEncoder()
yencoder.fit(Y)
encoded_Y = yencoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
hot_y = np_utils.to_categorical(encoded_Y)
pd.DataFrame(hot_y).head()

# Encode string features
columns = ["Gender", "Ever_Married",
           "Graduated", "Profession", "Spending_Score"]
for feature in columns:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])

df = df.drop(["Segmentation", "ID"], axis=1)
df.head()


def fillmissing(df, feature, method):
    if method == "mode":
        df[feature] = df[feature].fillna(df[feature].mode()[0])
    elif method == "median":
        df[feature] = df[feature].fillna(df[feature].median())
    else:
        df[feature] = df[feature].fillna(df[feature].mean())


features_missing = df.columns[df.isna().any()]
for feature in features_missing:
    fillmissing(df, feature=feature, method="mean")

df.info()

X = df.to_numpy()
mu = X.mean(0)
sigma = X.std(0)  # standard deviation: max(x)-min(x)
xn = (X - mu) / sigma

# define baseline model


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(18, input_dim=8, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model,
                            epochs=200, batch_size=100, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, xn, hot_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate the keras model
estimator.fit(X, Y)
pred = estimator.predict(X)

pd.DataFrame(pred)
