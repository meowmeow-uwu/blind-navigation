#!/usr/bin/env python3
"""Inspect ONNX model graphs and validate structure.

Checks:
- ONNX graph validity
- Input/output metadata
- Opset version
- Node count
- Dynamic vs static dimensions
- Unusual operators

Usage:
    python scripts/inspect_onnx.py --models-dir weights/onnx
    python scripts/inspect_onnx.py --models-dir weights/onnx --output weights/metadata/onnx_manifest.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx
import onnx.checker


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def inspect_onnx_model(onnx_path: Path) -> dict:
    """Inspect an ONNX model and return metadata.
    
    Returns:
        dict with model info, inputs, outputs, graph validity
    """
    try:
        rel_path = onnx_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = onnx_path
    
    result = {
        "model": onnx_path.stem,
        "path": str(rel_path),
        "file_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256_file(onnx_path),
        "graph_valid": False,
        "error": None,
        "opset": None,
        "inputs": [],
        "outputs": [],
        "node_count": 0,
        "op_types": [],
    }
    
    try:
        # Load model
        model = onnx.load(str(onnx_path))
        
        # Check model
        try:
            onnx.checker.check_model(model)
            result["graph_valid"] = True
        except Exception as e:
            result["error"] = f"ONNX checker failed: {str(e)}"
            result["graph_valid"] = False
        
        # Opset
        if model.opset_import:
            result["opset"] = model.opset_import[0].version
        
        # Inputs
        for inp in model.graph.input:
            input_info = {
                "name": inp.name,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            }
            
            # Shape
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)  # dynamic dimension
                else:
                    shape.append("?")
            
            input_info["shape"] = shape
            result["inputs"].append(input_info)
        
        # Outputs
        for out in model.graph.output:
            output_info = {
                "name": out.name,
            }
            
            # Try to get dtype
            try:
                output_info["dtype"] = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            except:
                output_info["dtype"] = "unknown"
            
            # Shape
            shape = []
            try:
                for dim in out.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            except:
                shape = ["unknown"]
            
            output_info["shape"] = shape
            result["outputs"].append(output_info)
        
        # Nodes
        result["node_count"] = len(model.graph.node)
        
        # Operator types
        op_types = set()
        for node in model.graph.node:
            op_types.add(node.op_type)
        
        result["op_types"] = sorted(list(op_types))
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["graph_valid"] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect ONNX models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("weights/onnx"),
        help="Directory containing ONNX files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (optional)",
    )
    
    args = parser.parse_args()
    
    if not args.models_dir.exists():
        print(f"Error: Directory not found: {args.models_dir}")
        sys.exit(1)
    
    # Find all ONNX files
    onnx_files = sorted(args.models_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"No ONNX files found in {args.models_dir}")
        sys.exit(1)
    
    print(f"Found {len(onnx_files)} ONNX file(s)")
    print(f"{'='*80}")
    
    # Inspect each model
    results = []
    
    for onnx_path in onnx_files:
        print(f"\nInspecting: {onnx_path.name}")
        print("-" * 80)
        
        result = inspect_onnx_model(onnx_path)
        results.append(result)
        
        # Print summary
        if result["graph_valid"]:
            print(f"✓ Graph valid")
        else:
            print(f"✗ Graph invalid: {result.get('error', 'unknown')}")
        
        print(f"File size: {result['file_size_mb']} MB")
        print(f"SHA256: {result['sha256'][:16]}...")
        print(f"Opset: {result['opset']}")
        print(f"Nodes: {result['node_count']}")
        
        print(f"\nInputs ({len(result['inputs'])}):")
        for inp in result["inputs"]:
            print(f"  - {inp['name']}: {inp['dtype']} {inp['shape']}")
        
        print(f"\nOutputs ({len(result['outputs'])}):")
        for out in result["outputs"]:
            print(f"  - {out['name']}: {out.get('dtype', '?')} {out['shape']}")
        
        print(f"\nOperator types ({len(result['op_types'])}):")
        print(f"  {', '.join(result['op_types'][:20])}")
        if len(result['op_types']) > 20:
            print(f"  ... and {len(result['op_types']) - 20} more")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    valid = [r for r in results if r["graph_valid"]]
    invalid = [r for r in results if not r["graph_valid"]]
    
    print(f"Valid:   {len(valid)}")
    print(f"Invalid: {len(invalid)}")
    
    if invalid:
        print("\n✗ Invalid models:")
        for r in invalid:
            print(f"  - {r['model']}: {r.get('error', 'unknown error')}")
    
    # Save to JSON
    if args.output:
        output_data = {
            "inspected_at": None,  # Will be set by datetime
            "models_dir": str(args.models_dir),
            "total_models": len(results),
            "valid_models": len(valid),
            "invalid_models": len(invalid),
            "models": {r["model"]: r for r in results},
        }
        
        from datetime import datetime
        output_data["inspected_at"] = datetime.utcnow().isoformat() + "Z"
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output}")
    
    # Exit code
    if invalid:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
