from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_precision: Literal["fp32", "int8"] = "int8"

    embedding_model_dir_fp32: str = "modelo_onnx"
    embedding_model_dir_int8: str = "modelo_onnx_int8"
    reranker_model_dir_fp32: str = "reranker_onnx_fp32"
    reranker_model_dir_int8: str = "reranker_onnx_int8"

    embedding_max_length: int = 2048
    reranker_max_length: int = 1024

    onnx_intra_op_num_threads: int = 0
    onnx_inter_op_num_threads: int = 0

    host: str = "0.0.0.0"
    port: int = 8001

    reranker_batch_size: int = 16

    # Use ["*"] for development; restrict to specific origins in production
    cors_origins: list[str] = ["*"]

    @property
    def embedding_model_dir(self) -> Path:
        d = self.embedding_model_dir_int8 if self.model_precision == "int8" else self.embedding_model_dir_fp32
        return Path(d)

    @property
    def reranker_model_dir(self) -> Path:
        d = self.reranker_model_dir_int8 if self.model_precision == "int8" else self.reranker_model_dir_fp32
        return Path(d)


settings = Settings()


def validate_model_dir(path: Path, label: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{label}: directory not found: {path.resolve()}")
    onnx_files = list(path.glob("*.onnx"))
    if len(onnx_files) == 0:
        raise RuntimeError(f"{label}: no .onnx file found in {path.resolve()}")
    if len(onnx_files) > 1:
        raise RuntimeError(f"{label}: multiple .onnx files found — ambiguous: {onnx_files}")
    return onnx_files[0]
