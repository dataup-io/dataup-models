from pydantic import BaseModel, Field
from typing import Literal
from dataup_models.geom import BoundingBox
from dataup_models.labels import Label


class DetectorParams(BaseModel):
    param_type: Literal["detector"] = "detector"
    threshold: float = Field(
        description="Confidence threshold", ge=0.0, le=1.0, default=0.5
    )
    iou_threshold: float = Field(description="IoU threshold", ge=0.0, le=1.0, default=0)
    max_detections: int = Field(
        description="Max detections per image", ge=1, le=1000, default=100
    )


class SAMParams(BaseModel):
    param_type: Literal["sam2"] = "sam2"
    pos_points: list[tuple[int, int]] | None = None
    neg_points: list[tuple[int, int]] | None = None
    pos_boxes: list[BoundingBox] | None = None


class DetectionResults(BaseModel):
    image_id: str = Field(description="Image ID")
    labels: list[Label] = Field(description="List of detected labels")

    @classmethod
    def from_rbf(cls, image_id: str, rbf_preds: list[dict]) -> "DetectionResults":
        labels = [Label.from_rbf(pred) for pred in rbf_preds]
        return cls(image_id=image_id, labels=labels)


class SAMblob(BaseModel):
    image_id: str = Field(description="Image ID")
    blob: str = Field(description="SAM2 blob")
    shape: tuple[int, ...] = Field(description="Shape of the blob")
    dtype: str = Field(description="Data type of the blob")
