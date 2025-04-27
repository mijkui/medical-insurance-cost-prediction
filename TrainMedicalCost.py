# %% -------------------- 1. Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# %% -------------------- 2. Load and inspect data
df = pd.read_csv("medical_insurance.csv")

# quick sanity check
print(df.head(), "\n")
print(df.info(), "\n")

# %% -------------------- 3. Feature / target split
y = df["charges"]
X = df.drop(columns="charges")

num_cols  = ["age", "bmi", "children"]
cat_cols  = ["sex", "smoker", "region"]

# %% -------------------- 4. Pre-processing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# %% -------------------- 5. Train / validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df["smoker"]
)

# %% -------------------- 6. Fit transformer **first**, then transform
preprocess.fit(X_train)
X_train_t = preprocess.transform(X_train)
X_valid_t = preprocess.transform(X_valid)

feature_names = (
    list(preprocess.named_transformers_["num"].get_feature_names_out(num_cols))
    + list(preprocess.named_transformers_["cat"].get_feature_names_out(cat_cols))
)

# %% -------------------- 7. Model training
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_t, y_train)

# %% -------------------- 8. Quick validation
pred_valid = rf.predict(X_valid_t)
print(f"MAE : {mean_absolute_error(y_valid, pred_valid):,.0f}")
print(f"R²  : {r2_score(y_valid, pred_valid):.3f}")

# %% -------------------- 9. Export artefacts
out = Path(".")          # change if you want a different folder
pickle.dump(preprocess,    open(out / "preprocess.pkl",   "wb"))
pickle.dump(feature_names, open(out / "feature_names.pkl","wb"))
pickle.dump(rf,            open(out / "rf_model.pkl",     "wb"))

print("\n✅ 3 files written: preprocess.pkl  feature_names.pkl  rf_model.pkl")
