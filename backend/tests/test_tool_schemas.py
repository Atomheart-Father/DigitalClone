"""
Tests for tool schema validation.

This module tests that all tool schemas are valid JSON Schema
and conform to our requirements.
"""

import pytest
import jsonschema
import sys
import os

# Add the backend directory to Python path for imports
backend_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_dir)

from tool_registry import registry
from tool_prompt_builder import build_tool_prompts


def test_all_tools_have_valid_schemas():
    """Test that all registered tools have valid JSON schemas."""
    tools = registry.list_tools()

    assert len(tools) > 0, "Should have at least one tool registered"

    for tool in tools:
        # Test schema validity
        try:
            jsonschema.Draft7Validator.check_schema(tool.parameters)
        except jsonschema.SchemaError as e:
            pytest.fail(f"Tool {tool.name} has invalid JSON schema: {e}")


def test_tool_schema_structure():
    """Test that tool schemas have required structure."""
    tools = registry.list_tools()

    for tool in tools:
        schema = tool.parameters

        # Must have type field
        assert "type" in schema, f"Tool {tool.name} missing 'type' in schema"
        assert schema["type"] == "object", f"Tool {tool.name} schema type must be 'object'"

        # Must have properties field
        assert "properties" in schema, f"Tool {tool.name} missing 'properties' in schema"
        assert isinstance(schema["properties"], dict), f"Tool {tool.name} properties must be a dict"

        # Must have required field
        assert "required" in schema, f"Tool {tool.name} missing 'required' field"
        assert isinstance(schema["required"], list), f"Tool {tool.name} required must be a list"


def test_tool_schema_additional_properties():
    """Test that tool schemas handle additional properties correctly."""
    tools = registry.list_tools()

    for tool in tools:
        schema = tool.parameters

        # Should have additionalProperties set to false for strict validation
        assert "additionalProperties" in schema, f"Tool {tool.name} should specify additionalProperties"
        assert schema["additionalProperties"] is False, f"Tool {tool.name} should not allow additional properties"


def test_calculator_tool_schema():
    """Test calculator tool schema specifically."""
    tools = registry.list_tools()
    calculator_tool = next((t for t in tools if t.name == "calculator"), None)

    assert calculator_tool is not None, "Calculator tool should be registered"

    schema = calculator_tool.parameters

    # Check required parameters
    assert "expression" in schema["required"], "Calculator should require expression parameter"

    # Check parameter definition
    assert "expression" in schema["properties"], "Calculator should define expression parameter"
    expr_prop = schema["properties"]["expression"]

    assert expr_prop["type"] == "string", "Expression should be string type"
    assert "description" in expr_prop, "Expression should have description"


def test_datetime_tool_schema():
    """Test datetime tool schema specifically."""
    tools = registry.list_tools()
    datetime_tool = next((t for t in tools if t.name == "datetime"), None)

    assert datetime_tool is not None, "Datetime tool should be registered"

    schema = datetime_tool.parameters

    # Check optional parameters
    properties = schema["properties"]
    assert "format" in properties, "Datetime should support format parameter"
    assert "tz" in properties, "Datetime should support timezone parameter"

    # Check format enum values
    format_prop = properties["format"]
    assert "enum" in format_prop, "Format should have enum values"
    expected_formats = ["date", "time", "datetime", "iso", "timestamp"]
    assert set(format_prop["enum"]) == set(expected_formats), "Format enum should match expected values"

    # Check timezone enum values
    tz_prop = properties["tz"]
    assert "enum" in tz_prop, "Timezone should have enum values"
    expected_timezones = ["local", "utc"]
    assert set(tz_prop["enum"]) == set(expected_timezones), "Timezone enum should match expected values"


def test_tool_schema_validation_with_examples():
    """Test schema validation with example inputs."""
    tool_prompts = build_tool_prompts()

    # Test calculator with valid input
    calculator_func = next((f for f in tool_prompts["tools"] if f["function"]["name"] == "calculator"), None)
    assert calculator_func is not None, "Calculator function should be in tools"

    schema = calculator_func["function"]["parameters"]

    # Valid input should pass
    valid_input = {"expression": "2 + 3 * 4"}
    jsonschema.validate(valid_input, schema)

    # Invalid input should fail
    with pytest.raises(jsonschema.ValidationError):
        invalid_input = {"wrong_param": "value"}
        jsonschema.validate(invalid_input, schema)

    # Missing required parameter should fail
    with pytest.raises(jsonschema.ValidationError):
        empty_input = {}
        jsonschema.validate(empty_input, schema)


def test_tool_name_uniqueness():
    """Test that all tool names are unique."""
    tools = registry.list_tools()
    names = [tool.name for tool in tools]

    assert len(names) == len(set(names)), "Tool names should be unique"


def test_tool_schema_completeness():
    """Test that schemas are complete and well-formed."""
    tools = registry.list_tools()

    for tool in tools:
        schema = tool.parameters

        # Check that all properties have descriptions
        for prop_name, prop_def in schema["properties"].items():
            assert "description" in prop_def, f"Property {prop_name} in tool {tool.name} should have description"

        # Check that required fields are actually defined
        for required_field in schema["required"]:
            assert required_field in schema["properties"], f"Required field {required_field} not defined in tool {tool.name}"
