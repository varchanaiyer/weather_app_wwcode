from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(df):
    X = df.drop("Temperature", axis=1)
    y = df["Temperature"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = DecisionTreeRegressor(random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions, y_test

def evaluate_model(predictions, y_test):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2
