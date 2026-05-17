from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


app = FastAPI(title="Linear Regression Prediction API")
BASE_DIR = Path(__file__).resolve().parent


# ==============================
# CORS CONFIGURATION
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# GLOBAL MODEL STORAGE
# ==============================

model_pipeline = None
feature_columns = None
target_name = None


# ==============================
# REQUEST MODEL FOR PREDICTION
# ==============================

class PredictionInput(BaseModel):
    features: dict


def preprocess_dataset(data: pd.DataFrame, target_column: str):
    cleaned = data.copy()
    original_rows, original_columns = cleaned.shape

    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    duplicate_columns = cleaned.columns[cleaned.columns.duplicated()].tolist()
    if duplicate_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate column names found after cleanup: {duplicate_columns}"
        )

    normalized_target = target_column.strip()
    if normalized_target not in cleaned.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist in dataset."
        )

    object_columns = cleaned.select_dtypes(include=["object"]).columns
    if len(object_columns) > 0:
        cleaned[object_columns] = cleaned[object_columns].apply(
            lambda column: column.map(lambda value: value.strip() if isinstance(value, str) else value)
        )
        cleaned[object_columns] = cleaned[object_columns].replace(r"^\s*$", np.nan, regex=True)

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    empty_row_count = int(cleaned.isna().all(axis=1).sum())
    if empty_row_count:
        cleaned = cleaned.dropna(axis=0, how="all")

    empty_columns = cleaned.columns[cleaned.isna().all(axis=0)].tolist()
    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)
        if normalized_target not in cleaned.columns:
            raise HTTPException(
                status_code=400,
                detail="Target column became empty after preprocessing."
            )

    duplicate_row_count = int(cleaned.duplicated().sum())
    if duplicate_row_count:
        cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.dropna(subset=[normalized_target])
    if cleaned.empty:
        raise HTTPException(
            status_code=400,
            detail="No usable rows remain after preprocessing the dataset."
        )

    target_series = pd.to_numeric(cleaned[normalized_target], errors="coerce")
    invalid_target_count = int(target_series.isna().sum())
    if invalid_target_count:
        cleaned = cleaned.loc[target_series.notna()].copy()
        target_series = target_series.loc[target_series.notna()]

    if cleaned.empty:
        raise HTTPException(
            status_code=400,
            detail="The target column has no valid numeric values after preprocessing."
        )

    cleaned[normalized_target] = target_series.astype(float)

    feature_frame = cleaned.drop(columns=[normalized_target]).copy()
    if feature_frame.empty:
        raise HTTPException(
            status_code=400,
            detail="The dataset must include at least one feature column besides the target."
        )

    for column in feature_frame.select_dtypes(include=["object"]).columns:
        numeric_candidate = pd.to_numeric(feature_frame[column], errors="coerce")
        non_null_mask = feature_frame[column].notna()
        if non_null_mask.any() and numeric_candidate.loc[non_null_mask].notna().all():
            feature_frame[column] = numeric_candidate

    cleaned[feature_frame.columns] = feature_frame

    if len(cleaned) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two usable rows are required after preprocessing."
        )

    preprocessing_summary = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "final_rows": int(len(cleaned)),
        "final_columns": int(cleaned.shape[1]),
        "duplicate_rows_removed": duplicate_row_count,
        "empty_rows_removed": empty_row_count,
        "empty_columns_removed": empty_columns,
        "invalid_target_rows_removed": invalid_target_count
    }

    return cleaned, normalized_target, preprocessing_summary


def build_visualization_payload(
    X: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str]
):
    plot_frame = pd.DataFrame({
        "actual": y_test.to_numpy(dtype=float),
        "predicted": np.asarray(y_pred, dtype=float)
    }).round(4)

    plot_frame = plot_frame.replace([np.inf, -np.inf], np.nan).dropna()
    plot_sample = plot_frame.head(60)

    numeric_summary = []
    for feature in numeric_features[:6]:
        series = pd.to_numeric(X[feature], errors="coerce").dropna()
        if series.empty:
            continue
        numeric_summary.append({
            "name": feature,
            "mean": round(float(series.mean()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4)
        })

    categorical_summary = []
    for feature in categorical_features[:4]:
        counts = X[feature].fillna("Missing").astype(str).value_counts().head(5)
        categorical_summary.append({
            "name": feature,
            "top_values": [
                {"label": str(label), "count": int(count)}
                for label, count in counts.items()
            ]
        })

    return {
        "prediction_scatter": plot_sample.to_dict(orient="records"),
        "feature_summary": {
            "numeric": numeric_summary,
            "categorical": categorical_summary
        }
    }


# ==============================
# HOME ROUTE
# ==============================

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "message": "Linear Regression API is running successfully."
    }


# ==============================
# TRAIN MODEL ROUTE
# ==============================

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    test_size: float = Form(0.2)
):
    global model_pipeline, feature_columns, target_name

    try:
        if not 0 < test_size < 1:
            raise HTTPException(
                status_code=400,
                detail="test_size must be a number between 0 and 1."
            )

        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))
        data, target_column, preprocessing_summary = preprocess_dataset(data, target_column)

        X = data.drop(columns=[target_column])
        y = data[target_column]

        feature_columns = list(X.columns)
        target_name = target_column

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean"))
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features)
            ]
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression())
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        visualization = build_visualization_payload(
            X,
            y_test,
            y_pred,
            numeric_features,
            categorical_features
        )

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        return {
            "message": "Model trained successfully.",
            "target_column": target_column,
            "features": feature_columns,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
            "preprocessing": preprocessing_summary,
            "visualization": visualization
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==============================
# PREDICTION ROUTE
# ==============================

@app.post("/predict")
def predict(input_data: PredictionInput):
    global model_pipeline, feature_columns

    if model_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="Model has not been trained yet. Please train the model first."
        )

    try:
        user_features = input_data.features

        prediction_df = pd.DataFrame([user_features])

        for col in feature_columns:
            if col not in prediction_df.columns:
                prediction_df[col] = np.nan

        prediction_df = prediction_df[feature_columns]

        prediction = model_pipeline.predict(prediction_df)

        return {
            "prediction": round(float(prediction[0]), 4)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
