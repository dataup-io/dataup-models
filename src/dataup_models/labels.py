import hashlib
from dataup_models.geom import BoundingBox, Polygon
from pydantic import BaseModel


class LabelAttribute(BaseModel, frozen=True):
    key: str
    value: str


class Label(BaseModel, frozen=True):
    label: str
    score: float
    bbox: BoundingBox
    polygon: Polygon | None = None
    rle_mask: str = ""
    attributes: list[LabelAttribute] = []

    @property
    def hash(self) -> str:
        bbox = self.bbox if self.polygon is None else self.polygon.to_bbox()
        hash_input = f"{self.label}:{bbox.x}:{bbox.y}:{bbox.width}:{bbox.height}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def add_attributes(
        self, *attributes: LabelAttribute, replace: bool = False
    ) -> "Label":
        existing_attributes = self.attributes
        if replace:
            keys_to_replace = {attr.key for attr in attributes}
            existing_attributes = [
                attr for attr in existing_attributes if attr.key not in keys_to_replace
            ]
        return self.model_copy(
            update={"attributes": [*existing_attributes, *attributes]}, deep=True
        )

    def get_attribute(self, key: str) -> LabelAttribute | None:
        for attr in self.attributes:
            if attr.key == key:
                return attr
        return None
