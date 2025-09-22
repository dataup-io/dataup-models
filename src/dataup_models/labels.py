import hashlib
from dataup_models.geom import BoundingBox, Polygon
from pydantic import BaseModel


class LabelAttribute(BaseModel, frozen=True):
    name: str
    value: str

class Label(BaseModel, frozen=True):
    label: str
    score: float
    bbox: BoundingBox
    polygon: Polygon | None = None
    rle_mask: str = ""
    attributes: list[LabelAttribute] | None = None

    @property
    def hash(self) -> str:
        bbox = self.bbox if self.polygon is None else self.polygon.to_bbox()
        hash_input = f"{self.label}:{bbox.x}:{bbox.y}:{bbox.width}:{bbox.height}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]