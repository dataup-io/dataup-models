from pydantic import BaseModel, Field

class BoundingBox(BaseModel, frozen=True):
    x: int
    y: int
    width: int
    height: int

class Polygon(BaseModel, frozen=True):
    points: list[tuple[int, int]] = Field(description="List of points in the polygon")

    def to_bbox(self) -> BoundingBox:
        return BoundingBox(
            x=self.x_min,
            y=self.y_min,
            width=self.x_max - self.x_min,
            height=self.y_max - self.y_min,
        )

    @classmethod
    def from_bbox(cls, bbox: BoundingBox) -> "Polygon":
        return cls(
            points=[
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (bbox.x, bbox.y + bbox.height),
            ]
        )

    @property
    def x_min(self) -> int:
        return min(x for x, y in self.points)

    @property
    def y_min(self) -> int:
        return min(y for x, y in self.points)

    @property
    def x_max(self) -> int:
        return max(x for x, y in self.points)

    @property
    def y_max(self) -> int:
        return max(y for x, y in self.points)
