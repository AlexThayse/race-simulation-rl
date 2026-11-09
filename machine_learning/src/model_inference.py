"""Inference for new PyTorch policies and the bundled legacy TFLite models."""

from pathlib import Path
import numpy as np


class ModelInterpreter:
    """Expose the small interpreter interface used by the existing VSE classes."""

    def __init__(self, model_path):
        self._torch_model = None
        self._lite = None
        if Path(model_path).suffix == '.pt':
            from machine_learning.src.q_network import load_q_network
            self._torch_model = load_q_network(model_path)
        else:
            from ai_edge_litert.interpreter import Interpreter
            self._lite = Interpreter(model_path=str(model_path))

    def allocate_tensors(self):
        if self._lite is not None:
            self._lite.allocate_tensors()

    def get_input_details(self):
        if self._lite is not None:
            return self._lite.get_input_details()
        return [{'index': 0}]

    def get_output_details(self):
        if self._lite is not None:
            return self._lite.get_output_details()
        return [{'index': 0, 'shape': [1, self._torch_model.no_actions]}]

    def set_tensor(self, index, value):
        if self._lite is not None:
            self._lite.set_tensor(index, value)
        else:
            self._input = np.asarray(value, dtype=np.float32)

    def invoke(self):
        if self._lite is not None:
            self._lite.invoke()
        else:
            import torch
            with torch.inference_mode():
                self._output = self._torch_model(torch.from_numpy(self._input)).numpy()

    def get_tensor(self, index):
        if self._lite is not None:
            return self._lite.get_tensor(index)
        return self._output
