from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class AmparaReranker:
    def __init__(
        self,
        model_dir: Path,
        max_length: int,
        batch_size: int,
        session_options: ort.SessionOptions,
    ):
        onnx_files = list(model_dir.glob("*.onnx"))
        self.session = ort.InferenceSession(
            str(onnx_files[0]),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.max_length = max_length
        self.batch_size = batch_size
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def compute_scores(self, query: str, documents: list[str]) -> np.ndarray:
        pairs = [[query, doc] for doc in documents]
        all_logits: list[np.ndarray] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {k: v for k, v in encoded.items() if k in self._input_names}
            outputs = self.session.run(None, feed)
            all_logits.append(outputs[0].flatten())
        logits = np.concatenate(all_logits)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
