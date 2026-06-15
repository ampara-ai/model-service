from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class AmparaEmbedder:
    def __init__(self, model_dir: Path, max_length: int, session_options: ort.SessionOptions):
        onnx_files = list(model_dir.glob("*.onnx"))
        self.session = ort.InferenceSession(
            str(onnx_files[0]),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.max_length = max_length
        self._input_names = {inp.name for inp in self.session.get_inputs()}
        # Output shape is [batch, seq_len, hidden_size]; hidden_size is statically 1024 for BGE-M3
        output_shape = self.session.get_outputs()[0].shape
        self.dimension: int = output_shape[-1] if output_shape[-1] is not None else 1024

    def encode(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {k: v for k, v in encoded.items() if k in self._input_names}
        outputs = self.session.run(None, feed)
        cls = outputs[0][:, 0, :]
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-8, norm)
        return (cls / norm).astype(np.float32).squeeze(0)
