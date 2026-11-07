"""Tests for ONNX model graphs."""

import json
from pathlib import Path

import onnx
import onnx.checker
import pytest


def get_onnx_models():
    """Get list of exported ONNX models."""
    manifest_path = Path("weights/metadata/onnx_manifest.json")
    if not manifest_path.exists():
        pytest.skip("ONNX manifest not found - run export first")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    models = []
    for model_name, model_info in manifest["models"].items():
        if model_info.get("graph_valid"):
            models.append((model_name, Path(model_info["path"])))
    
    if not models:
        pytest.skip("No valid ONNX models found")
    
    return models


@pytest.mark.parametrize("model_name,onnx_path", get_onnx_models())
def test_onnx_graph_is_valid(model_name, onnx_path):
    """Test that ONNX graph passes onnx.checker."""
    if not onnx_path.exists():
        pytest.skip(f"ONNX file not found: {onnx_path}")
    
    model = onnx.load(str(onnx_path))
    
    # Should not raise
    onnx.checker.check_model(model)


@pytest.mark.parametrize("model_name,onnx_path", get_onnx_models())
def test_input_shape_is_static_320(model_name, onnx_path):
    """Test that ONNX model has static 320x320 input."""
    if not onnx_path.exists():
        pytest.skip(f"ONNX file not found: {onnx_path}")
    
    model = onnx.load(str(onnx_path))
    
    # Get first input
    input_tensor = model.graph.input[0]
    
    # Check shape: [1, 3, 320, 320]
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append(None)  # Dynamic dimension
    
    assert shape == [1, 3, 320, 320], f"{model_name} has unexpected input shape: {shape}"


@pytest.mark.parametrize("model_name,onnx_path", get_onnx_models())
def test_input_dtype_is_fp32(model_name, onnx_path):
    """Test that ONNX model expects FP32 input."""
    if not onnx_path.exists():
        pytest.skip(f"ONNX file not found: {onnx_path}")
    
    model = onnx.load(str(onnx_path))
    
    input_tensor = model.graph.input[0]
    dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
    
    assert dtype == "FLOAT", f"{model_name} has unexpected input dtype: {dtype}"


@pytest.mark.parametrize("model_name,onnx_path", get_onnx_models())
def test_opset_version(model_name, onnx_path):
    """Test that ONNX model uses expected opset."""
    if not onnx_path.exists():
        pytest.skip(f"ONNX file not found: {onnx_path}")
    
    model = onnx.load(str(onnx_path))
    
    if model.opset_import:
        opset = model.opset_import[0].version
        # We export with opset 12
        assert opset == 12, f"{model_name} has opset {opset}, expected 12"
