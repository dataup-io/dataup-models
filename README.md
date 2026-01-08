# dataup-models

**DataUp Models** is a Python library that defines the data contract for integrating custom and private machine learning models with [cvat-dataup](https://github.com/dataup-io/cvat-dataup). This library is open-sourced to provide clear documentation of the data schemas and types required for seamless integration.

Built with Pydantic v2, DataUp Models ensures type safety, validation, and serialization for all inference requests and responses, making it easy to connect your models to the DataUp platform.

## Purpose

DataUp Models serves as the **official data contract** for integrating with cvat-dataup. By using these models in your custom or private ML model implementations, you ensure:

- ✅ **Compatibility** with the DataUp platform
- ✅ **Type safety** and validation for all data exchanges
- ✅ **Clear documentation** of expected request/response formats
- ✅ **Seamless integration** without worrying about data format mismatches

## Features

- 🎯 **Type-safe models** for ML inference requests and responses
- 📦 **Geometric primitives** for bounding boxes and polygons
- 🏷️ **Label management** with attributes and metadata
- ✅ **Pydantic validation** ensuring data integrity
- 🧊 **Immutable models** for thread-safe operations
- 🔌 **cvat-dataup integration** ready out of the box

## Installation

### Using `uv` (Recommended)

```bash
uv pip install -e .
```

### Using `pip`

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- Pydantic >= 2.10.6

## Integration with cvat-dataup

To integrate your custom or private ML model with cvat-dataup, use the models defined in this library for all data exchanges:

1. **Import the models** in your inference service
2. **Use `InferenceRequest`** to parse incoming requests from DataUp
3. **Return `InferenceResponse`** with your model's predictions
4. **Convert your model outputs** to `DetectionResults` and `Label` objects

This ensures your service speaks the same "language" as cvat-dataup, enabling seamless integration.

### Example Integration

```python
from dataup_models.requests import InferenceRequest, InferenceResponse
from dataup_models.ml import DetectionResults, DetectorParams
from dataup_models.labels import Label
from dataup_models.geom import BoundingBox

def run_inference(request: InferenceRequest) -> InferenceResponse:
    """Your custom inference function"""
    # Parse the request (already validated by Pydantic)
    image_urls = request.image_urls
    params = request.params
    
    # Run your model inference
    # ... your model code here ...
    
    # Convert your model outputs to DataUp format
    labels = [
        Label(
            label="person",
            score=0.95,
            bbox=BoundingBox(x=10, y=20, width=100, height=150)
        )
    ]
    
    results = DetectionResults(
        image_id="img-001",
        labels=labels
    )
    
    # Return in the expected format
    return InferenceResponse(
        data=[results],
        success=True,
        session_id=request.session_id
    )
```

## Quick Start

### Basic Usage

```python
from dataup_models.geom import BoundingBox, Polygon
from dataup_models.labels import Label, LabelAttribute
from dataup_models.ml import DetectorParams, DetectionResults
from dataup_models.requests import InferenceRequest, InferenceResponse

# Create a bounding box
bbox = BoundingBox(x=10, y=20, width=100, height=150)

# Create a label with attributes
label = Label(
    label="person",
    score=0.95,
    bbox=bbox,
    attributes=[
        LabelAttribute(key="age", value="25"),
        LabelAttribute(key="gender", value="male")
    ]
)

# Create model parameters
params = DetectorParams(
    threshold=0.5,
    iou_threshold=0.4,
    max_detections=100
)

# Create an inference request
request = InferenceRequest(
    image_urls=["https://example.com/image.jpg"],
    params=params,
    session_id="session-123"
)

# Create detection results
results = DetectionResults(
    image_id="img-001",
    labels=[label]
)

# Create an inference response
response = InferenceResponse(
    data=[results],
    success=True,
    session_id="session-123"
)
```

## API Reference

### Geometric Models (`geom.py`)

#### `BoundingBox`

Represents a rectangular bounding box with integer coordinates.

```python
class BoundingBox(BaseModel, frozen=True):
    x: int          # Top-left x coordinate
    y: int          # Top-left y coordinate
    width: int      # Box width
    height: int     # Box height
```

**Example:**
```python
bbox = BoundingBox(x=10, y=20, width=100, height=150)
```

#### `Polygon`

Represents a polygon as a list of integer coordinate points.

```python
class Polygon(BaseModel, frozen=True):
    points: list[tuple[int, int]]  # List of (x, y) coordinate pairs
```

**Methods:**
- `to_bbox() -> BoundingBox`: Convert polygon to its bounding box
- `from_bbox(bbox: BoundingBox) -> Polygon`: Create a rectangular polygon from a bounding box

**Properties:**
- `x_min`, `y_min`, `x_max`, `y_max`: Bounding box boundaries

**Example:**
```python
polygon = Polygon(points=[(0, 0), (100, 0), (100, 100), (0, 100)])
bbox = polygon.to_bbox()
```

### Label Models (`labels.py`)

#### `Label`

Represents a detected object with its metadata.

```python
class Label(BaseModel, frozen=True):
    label: str                      # Class name
    score: float                    # Confidence score (0.0-1.0)
    bbox: BoundingBox              # Bounding box
    polygon: Polygon | None         # Optional polygon segmentation
    rle_mask: str                  # Optional RLE-encoded mask
    attributes: list[LabelAttribute]  # Custom attributes
```

**Methods:**
- `hash: str` (property): Unique hash based on label and bounding box
- `add_attributes(*attributes, replace=False) -> Label`: Add or replace attributes
- `get_attribute(key: str) -> LabelAttribute | None`: Get attribute by key

**Example:**
```python
label = Label(
    label="cat",
    score=0.92,
    bbox=BoundingBox(x=50, y=60, width=80, height=100),
    attributes=[
        LabelAttribute(key="breed", value="persian")
    ]
)

# Add more attributes
label = label.add_attributes(
    LabelAttribute(key="color", value="white"),
    replace=False
)

# Get attribute
breed = label.get_attribute("breed")
```

#### `LabelAttribute`

Key-value pair for label metadata.

```python
class LabelAttribute(BaseModel, frozen=True):
    key: str
    value: str
```

### ML Models (`ml.py`)

#### `DetectorParams`

Parameters for object detection models.

```python
class DetectorParams(BaseModel):
    param_type: Literal["detector"] = "detector"
    threshold: float = 0.5          # Confidence threshold (0.0-1.0)
    iou_threshold: float = 0.0     # IoU threshold for NMS (0.0-1.0)
    max_detections: int = 100       # Maximum detections per image (1-1000)
    prompt: str | None = None      # Optional text prompt
```

#### `SAM3Params`

Parameters for SAM3 (Segment Anything Model 3) inference.

```python
class SAM3Params(BaseModel):
    param_type: Literal["sam3"] = "sam3"
    text_prompt: str | None = None        # Text prompt for segmentation
    geom_prompt: dict | None = None       # Geometric prompt (points, boxes, etc.)
```

#### `DetectionResults`

Container for detection results from a single image.

```python
class DetectionResults(BaseModel):
    image_id: str
    labels: list[Label]
```

**Example:**
```python
from dataup_models.ml import DetectionResults
from dataup_models.labels import Label
from dataup_models.geom import BoundingBox

results = DetectionResults(
    image_id="img-001",
    labels=[
        Label(
            label="person",
            score=0.95,
            bbox=BoundingBox(x=10, y=20, width=80, height=200)
        )
    ]
)
```

### Request/Response Models (`requests.py`)

#### `InferenceRequest`

Request model for ML inference.

```python
class InferenceRequest(BaseModel, frozen=True):
    request_id: str | None = None
    image_urls: list[str] | None = None      # List of image URLs
    images_b64: list[HttpUrl] | None = None  # List of base64-encoded images
    session_id: str | None = None
    params: ModelParams                      # DetectorParams | SAM3Params
```

**Example:**
```python
request = InferenceRequest(
    image_urls=["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
    params=DetectorParams(threshold=0.6, max_detections=50),
    session_id="session-123"
)
```

#### `InferenceResponse`

Response model for ML inference.

```python
class InferenceResponse(BaseModel):
    data: list[DetectionResults] | dict     # Results or custom data
    success: bool = True
    session_id: str | None = None
    error: str = ""
```

**Example:**
```python
response = InferenceResponse(
    data=[detection_results],
    success=True,
    session_id="session-123"
)
```

## Advanced Usage

### Working with Discriminated Unions

The `ModelParams` type uses Pydantic's discriminated union feature, allowing automatic validation based on the `param_type` field:

```python
from dataup_models.requests import InferenceRequest
from dataup_models.ml import DetectorParams, SAM3Params

# DetectorParams will be automatically selected
request1 = InferenceRequest(
    image_urls=["https://example.com/image.jpg"],
    params=DetectorParams(threshold=0.7)
)

# SAM3Params will be automatically selected
request2 = InferenceRequest(
    image_urls=["https://example.com/image.jpg"],
    params=SAM3Params(text_prompt="a red car")
)
```

### Immutable Models

All models are frozen (immutable), ensuring thread-safety. To modify, use `model_copy()`:

```python
label = Label(
    label="cat",
    score=0.9,
    bbox=BoundingBox(x=0, y=0, width=100, height=100)
)

# Create a new label with updated score
new_label = label.model_copy(update={"score": 0.95})
```

## Development

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dataup-models
```

2. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

3. Install pre-commit hooks (if configured):
```bash
pre-commit install
```

### Project Structure

```
dataup-models/
├── src/
│   └── dataup_models/
│       ├── __init__.py
│       ├── geom.py          # Geometric primitives
│       ├── labels.py        # Label models
│       ├── ml.py            # ML model parameters and results
│       └── requests.py      # Request/response models
├── tests/
│   └── test_labels.py       # Unit tests
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dataup_models --cov-report=html

# Run specific test file
pytest tests/test_labels.py
```

### Code Quality

The project uses:
- **Ruff** for linting (line length: 150)
- **Pytest** for testing
- **Pydantic** for validation

### Type Checking

```bash
# Using mypy (if configured)
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

[Add your license here]

## Support

For issues, questions, or contributions related to:
- **Data contract questions**: Open an issue to clarify expected formats
- **Integration help**: Check the integration examples above or open a discussion
- **Bug reports**: Please include a minimal reproducible example
- **Feature requests**: We welcome suggestions for additional model support

---

## About This Library

**DataUp Models** is open-sourced to provide transparency about the data contract required for integrating with cvat-dataup. By using these models, you ensure your custom or private ML models can seamlessly communicate with the DataUp platform.

**Key Points:**
- This library defines the **official data contract** for cvat-dataup integration
- All models are **immutable (frozen)** for thread-safety
- Built with **Pydantic v2** for robust validation and serialization
- Requires **Python 3.10+** and leverages modern type hints

**Why Open Source?**
We believe in transparency and want to make it as easy as possible for developers to integrate their models with cvat-dataup. By open-sourcing the data models, you know exactly what format to use without needing to reverse-engineer the API.

