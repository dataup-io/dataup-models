from pydantic import BaseModel, HttpUrl, Field
from typing import Annotated
from dataup_models.ml import DetectorParams, SAM3Params, DetectionResults

ModelParams = Annotated[DetectorParams | SAM3Params, Field(discriminator="param_type")]


class InferenceRequest(BaseModel, frozen=True):
    request_id: str | None = Field(description="Unique request ID", default=None)
    image_urls: list[str] | None = Field(description="List of image URLs to run inference on", default=None)
    images_b64: list[str] | None = Field(description="List of base64 encoded images to run inference on", default=None)
    session_id: str | None = Field(description="Session ID", default=None)
    params: ModelParams = Field(default_factory=DetectorParams, description="Model inference parameters")


CoreServiceData = Annotated[list[DetectionResults] | dict, Field(...)]


class InferenceResponse(BaseModel):
    data: CoreServiceData
    success: bool = True
    session_id: str | None = Field(description="Session ID", default=None)
    error: str = ""
