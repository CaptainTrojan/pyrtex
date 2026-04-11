import pytest
from pydantic import BaseModel
from typing import Dict, Any, List

from pyrtex.client import create_model_from_schema, Job

def test_dynamic_schema_generation():
    """Ensure dynamic schema handles all branches properly."""
    schema_dict = {
        "title": "ComplexSchema",
        "type": "object",
        "$defs": {
            "Address": {
                "title": "Address",
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "integer"}
                },
                "required": ["city", "zip"]
            }
        },
        "properties": {
            "my_ref": {"$ref": "#/$defs/Address"},
            "my_number": {"type": "number", "default": 3.14},
            "my_array": {
                "type": "array",
                "items": {"type": "string"}
            },
            "my_dict_typed": {
                "type": "object",
                "additionalProperties": {"type": "integer"}
            },
            "my_dict_untyped": {
                "type": "object",
                "additionalProperties": True
            },
            "my_nested_object": {
                "type": "object",
                "properties": {
                    "inner": {"type": "boolean"}
                }
            },
            "unknown_type": {
                "type": "something_unknown"
            }
        },
        "required": ["my_ref", "my_array"]
    }

    DynamicModel = create_model_from_schema(schema_dict)

    instance = DynamicModel(
        my_ref={"city": "NYC", "zip": 10001},
        my_array=["a", "b"],
        my_dict_typed={"a": 1},
        my_dict_untyped={"x": 5},
        my_nested_object={"inner": True},
        unknown_type="anything"
    )

    assert instance.my_number == 3.14
    assert instance.my_array == ["a", "b"]
    assert instance.my_ref.city == "NYC"
    assert instance.my_ref.zip == 10001

def test_dummy_generation_with_enum():
    """Test mapping Enum in the dummy generation."""
    schema_dict = {
        "title": "EnumSchema",
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["A", "B", "C"]
            }
        },
        "required": ["category"]
    }
    DynamicModel = create_model_from_schema(schema_dict)

    class DummyInput(BaseModel):
        text: str

    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=DynamicModel,
        prompt_template="test",
        simulation_mode=True
    )
    job.add_request("req1", DummyInput(text="test"))
    
    # Generating results triggers _create_dummy_output which hits the Enum branch
    results = list(job.submit().wait().results())
    
    assert len(results) == 1
    assert results[0].was_successful
    assert results[0].output.category.value == "A"
