"""Inference API; load the packaged preprocessing and classifier once."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from pmldl.common import CLASSES, FEATURES, ROOT


class IrisInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    sepal_length: float = Field(gt=0, le=30, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=30, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=30, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=30, description="Petal width in cm")


class Prediction(BaseModel):
    species: str
    probabilities: dict[str, float]
    model_run_id: str


def create_app(model_path: Path | None = None) -> FastAPI:
    path = model_path or Path(os.environ.get("MODEL_PATH", str(ROOT / "models/model.joblib")))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not path.is_file():
            raise RuntimeError(f"Model not found: {path}. Run the training stage first.")
        # joblib/pickle is unsafe for untrusted inputs. Only our training stage
        # provides this artifact; there is deliberately no model upload endpoint.
        bundle = joblib.load(path)
        if bundle.get("schema_version") != 1 or bundle["metadata"]["features"] != FEATURES:
            raise RuntimeError("Incompatible model artifact schema")
        if list(bundle["model"].classes_) != CLASSES:
            raise RuntimeError("Unexpected model classes")
        application.state.model = bundle["model"]
        application.state.metadata = bundle["metadata"]
        yield

    application = FastAPI(title="Iris prediction API", version="1.0.0", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_input(request, error: RequestValidationError):
        # Do not echo non-JSON values such as NaN/Infinity into an error response.
        details = [{key: item[key] for key in ("type", "loc", "msg")}
                   for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_run_id": application.state.metadata["run_id"]}

    @application.get("/model")
    def model_info() -> dict:
        return application.state.metadata

    @application.post("/predict", response_model=Prediction)
    def predict(sample: IrisInput) -> Prediction:
        frame = pd.DataFrame([sample.model_dump()], columns=FEATURES)
        model = application.state.model
        result = dict(zip(model.classes_, map(float, model.predict_proba(frame)[0])))
        return Prediction(species=max(result, key=result.get), probabilities=result,
                          model_run_id=application.state.metadata["run_id"])

    return application


app = create_app()
