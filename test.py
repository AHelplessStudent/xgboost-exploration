import xgboost as xgb
model_xgb_2 = xgb.XGBClassifier()
model_xgb_2.load_model("sample.json")
