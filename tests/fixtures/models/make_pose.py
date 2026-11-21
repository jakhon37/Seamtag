"""Writes pose.onnx, a tiny model shaped like RTMPose-m for tests: input "input" [N, 3, 256, 192],
outputs "simcc_x" [N, 17, 384] and "simcc_y" [N, 17, 512], two bins to an input pixel.

Each of COCO's 17 keypoints follows one colour of the input: the head's five red, the arms' six
green, the legs' six blue. Across, its values are that channel's mean down each column; down, its
mean along each row; so on black, each peaks at the middle of the disc of its colour. All are then
raised by 10, which gives each a positive score, except two: left_ear is lowered by 10 both ways,
and right_ear is raised by 20 across and lowered by 10 down, so the mean of its two highest values
is positive and the lower of them negative, which MMPose's scoring leaves out.
Run: uv run --with onnx python make_pose.py"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

here = Path(__file__).parent
H, W, K = 256, 192, 17
pick = np.zeros((K, 3), dtype=np.float32)
pick[np.arange(K), [0] * 5 + [1] * 6 + [2] * 6] = 1
raise_x = np.full((K, 1), 10, dtype=np.float32)
raise_y = np.full((K, 1), 10, dtype=np.float32)
raise_x[3], raise_y[3] = -10, -10
raise_x[4], raise_y[4] = 20, -10


def along(axis, length, name):
    """The channels' means along one axis of the image, each repeated for its two bins, one row per keypoint"""
    return [
        helper.make_node("ReduceMean", ["input"], [f"{name}_mean"], axes=[axis], keepdims=0),
        helper.make_node("Unsqueeze", [f"{name}_mean", "last"], [f"{name}_one"]),
        helper.make_node("Concat", [f"{name}_one", f"{name}_one"], [f"{name}_two"], axis=3),
        helper.make_node("Reshape", [f"{name}_two", f"{name}_shape"], [f"{name}_bins"]),
        helper.make_node("MatMul", ["pick", f"{name}_bins"], [f"{name}_picked"]),
        helper.make_node("Add", [f"{name}_picked", f"raise_{name}"], [f"simcc_{name}"]),
    ], numpy_helper.from_array(np.array([0, 0, 2 * length], dtype=np.int64), f"{name}_shape")


x_nodes, x_shape = along(2, W, "x")
y_nodes, y_shape = along(3, H, "y")
graph = helper.make_graph(
    x_nodes + y_nodes,
    "pose",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, H, W])],
    [
        helper.make_tensor_value_info("simcc_x", TensorProto.FLOAT, ["batch", K, 2 * W]),
        helper.make_tensor_value_info("simcc_y", TensorProto.FLOAT, ["batch", K, 2 * H]),
    ],
    [
        numpy_helper.from_array(pick, "pick"),
        numpy_helper.from_array(raise_x, "raise_x"),
        numpy_helper.from_array(raise_y, "raise_y"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "last"),
        x_shape,
        y_shape,
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.save(model, here / "pose.onnx")
