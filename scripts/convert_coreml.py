#!/usr/bin/env python3
"""Convert CoreML .mlmodelc to ONNX (runs on macOS in GitHub Actions)."""

import os
import sys

MODEL = os.environ.get("MODEL")
if not MODEL:
    print("ERROR: MODEL env var not set", file=sys.stderr)
    sys.exit(1)

import coremltools as ct

print(f"Loading CoreML model from: {MODEL}")
model = ct.models.MLModel(MODEL)

print("Input descriptions:")
for k, v in model.input_description.items():
    print(f"  {k}: {v}")

print("Output descriptions:")
for k, v in model.output_description.items():
    print(f"  {k}: {v}")

print("Converting to ONNX...")
onnx_model = ct.convert(model, target="onnx")

output_path = "inswapper_512_live.onnx"
onnx_model.save(output_path)
print(f"Saved ONNX to: {output_path}")

import onnx
m = onnx.load(output_path)
print("ONNX model loaded successfully")
print(f"  IR version: {m.ir_version}")
print(f"  Opset: {m.opset_import}")
print("Inputs:")
for inp in m.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape} ({inp.type.tensor_type.elem_type})")
print("Outputs:")
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape} ({out.type.tensor_type.elem_type})")
print("Conversion complete!")