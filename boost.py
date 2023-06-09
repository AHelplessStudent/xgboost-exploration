from xgboost import XGBClassifier
from xgboost import plot_tree
import xgboost as xgb
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], test_size=.2)


model2 = xgb.XGBClassifier()
model2.load_model("bst.json")
print("here")
# make predictions
preds = model2.predict(X_test)

print("yo")
print(preds)