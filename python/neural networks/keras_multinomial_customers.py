from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df = pd.read_csv('data/customertrain.csv')
df.head()

df.info()


def prepareY(df):
    # Drop rows with no Output values
    df.dropna(subset=['Var_1'], inplace=True)

    # extract Y and drop from dataframe
    Y = df["Var_1"]

    # encode class values as integers
    yencoder = LabelEncoder()
    yencoder.fit(Y)
    encoded_Y = yencoder.transform(Y)

    # convert integers to one hot encoded)
    return np_utils.to_categorical(encoded_Y), encoded_Y


hot_y, Y = prepareY(df)
df = df.drop(["Var_1"], axis=1)
pd.DataFrame(hot_y).head()


def fillmissing(df, feature, method):
    if method == "mode":
        df[feature] = df[feature].fillna(df[feature].mode()[0])
    elif method == "median":
        df[feature] = df[feature].fillna(df[feature].median())
    else:
        df[feature] = df[feature].fillna(df[feature].mean())


def prepareFeatures(df):
    # Encode string features to numerics
    columns = df.select_dtypes(include=['object']).columns
    # columns = ["Gender","Ever_Married","Graduated","Profession","Spending_Score"]
    for feature in columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    # fill in missing features with mean values
    features_missing = df.columns[df.isna().any()]
    for feature in features_missing:
        fillmissing(df, feature=feature, method="mean")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)


df = df.drop(["Segmentation", "ID"], axis=1)
X = prepareFeatures(df)
pd.DataFrame(X).head()

pd.DataFrame(X).info()


def baseline_model():
    # create model
    model = Sequential()
    # Rectified Linear Unit Activation Function
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # Softmax for multi-class classification
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X, hot_y, validation_split=0.33,
                    epochs=200, batch_size=100, verbose=0)

# evaluate the keras model
_, accuracy = model.evaluate(X, hot_y)
print('Accuracy: %.2f' % (accuracy*100))

predict_x = model.predict(X)
pred = np.argmax(predict_x, axis=1)
print(f'Prediction Accuracy: {(pred == Y).mean() * 100:f}')
