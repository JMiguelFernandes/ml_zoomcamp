import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data/bigmart_Train-Set.csv")

cols_categorical = ["FatContent", 
                    "ProductType", 
                    "OutletID"]
cols_numerical = ["ProductVisibility", 
                  "MRP"]

full_train, test = train_test_split(data[cols_categorical + cols_numerical + ["OutletSales"]], 
                                    test_size=0.2, 
                                    random_state=17)

y_full_train = np.log1p(full_train["OutletSales"])
y_test = np.log1p(test["OutletSales"])

del full_train["OutletSales"]
del test["OutletSales"]

dict_full_train = full_train.to_dict(orient="records")
dict_test = test.to_dict(orient="records")

dv=DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dict_full_train)
X_test = dv.transform(dict_test)

model = RandomForestRegressor(max_depth=6, 
                              min_samples_leaf=30, 
                              n_estimators=50,
                              random_state=17)

model.fit(X_full_train, y_full_train)

with open("model.bin", "wb") as file_out:
    pickle.dump(model, file_out)