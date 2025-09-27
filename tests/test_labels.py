from dataup_models.geom import BoundingBox
from dataup_models.labels import Label, LabelAttribute


def test_label_add_attributes():
    label = Label(
        label="cat", score=0.9, bbox=BoundingBox(x=0, y=0, width=100, height=100)
    )
    attr1 = LabelAttribute(key="attr1", value="attr1_value")
    attr2 = LabelAttribute(key="attr2", value="attr2_value")

    # Test replace=False
    label = label.add_attributes(attr1, attr2)
    assert label.attributes == [attr1, attr2]

    # Test replace=True
    attr1_replace = LabelAttribute(key="attr1", value="attr2_value")
    label = label.add_attributes(attr1_replace, replace=True)
    assert set(label.attributes) == set([attr1_replace, attr2])


def test_get_label_attribute():
    label = Label(
        label="cat", score=0.9, bbox=BoundingBox(x=0, y=0, width=100, height=100)
    )
    attr1 = LabelAttribute(key="attr1", value="attr1_value")
    attr2 = LabelAttribute(key="attr2", value="attr2_value")
    label = label.add_attributes(attr1, attr2)
    assert label.get_attribute("attr1") == attr1
    assert label.get_attribute("attr2") == attr2
    assert label.get_attribute("attr3") is None
