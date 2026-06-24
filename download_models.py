"""
Script de descarga de modelos desde Hugging Face Hub.
Ejecutado por entrypoint.sh antes de iniciar FastAPI.

Variables de entorno requeridas:
  HF_TOKEN        - Token de acceso a HF (repo privado)
  MODEL_PRECISION - "int8" (default) o "fp32"

Variables opcionales:
  MODELS_DIR      - Ruta donde se guardan los modelos (default: /app/models)
  FORCE_DOWNLOAD  - "true" para forzar re-descarga aunque ya existan
"""

import os
import sys
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────

REPO_ID        = "remii322/amparai-bge-m3-finetuned"
MODELS_DIR     = Path(os.environ.get("MODELS_DIR", "/app/models"))
HF_TOKEN       = os.environ.get("HF_TOKEN")
PRECISION      = os.environ.get("MODEL_PRECISION", "int8").lower()
FORCE_DOWNLOAD = os.environ.get("FORCE_DOWNLOAD", "false").lower() == "true"

# Qué carpetas descargar según la precisión elegida
FOLDERS_BY_PRECISION: dict[str, list[str]] = {
    "int8": [
        "modelo_onnx_int8",
        "reranker_onnx_int8",
        "modelo_base_onnx_int8",
    ],
    "fp32": [
        "modelo_onnx",
        "reranker_onnx_fp32",
        "modelo_base_onnx",
    ],
}

# Archivos que deben existir en cada carpeta de modelo para considerarla válida
REQUIRED_FILES_PER_FOLDER = [
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

# ─── Funciones ────────────────────────────────────────────────────────────────

def validate_env(*, require_hf_token: bool) -> None:
    """Valida variables requeridas para verificar o descargar modelos."""
    errors = []

    if require_hf_token and not HF_TOKEN:
        errors.append("HF_TOKEN no está configurado.")

    if PRECISION not in FOLDERS_BY_PRECISION:
        errors.append(
            f"MODEL_PRECISION='{PRECISION}' no es válido. "
            f"Debe ser uno de: {list(FOLDERS_BY_PRECISION.keys())}"
        )

    if errors:
        for msg in errors:
            log.error(msg)
        sys.exit(1)


def verify_models(folders: list[str]) -> bool:
    """
    Verifica que cada carpeta exista y contenga todos los archivos requeridos
    con tamaño mayor a 0. Retorna True solo si todo está correcto.
    """
    for folder in folders:
        folder_path = MODELS_DIR / folder

        if not folder_path.is_dir():
            log.warning(f"Carpeta no encontrada: {folder_path}")
            return False

        for filename in REQUIRED_FILES_PER_FOLDER:
            file_path = folder_path / filename

            if not file_path.exists():
                log.warning(f"Archivo faltante: {file_path}")
                return False

            if file_path.stat().st_size == 0:
                log.warning(f"Archivo vacío (descarga incompleta): {file_path}")
                return False

    return True


def download_models(folders: list[str]) -> None:
    """Descarga desde HF Hub solo las carpetas necesarias."""
    allow_patterns = [f"{folder}/*" for folder in folders]

    log.info(f"Carpetas a descargar: {folders}")
    log.info(f"Destino: {MODELS_DIR}")
    log.info("Iniciando descarga...")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODELS_DIR),
        allow_patterns=allow_patterns,
        token=HF_TOKEN,
        local_dir_use_symlinks=False,  # archivos reales, no symlinks — importante en Docker
    )


def get_marker_path() -> Path:
    """Ruta del archivo marker que indica que la descarga completó exitosamente."""
    return MODELS_DIR / f".ready_{PRECISION}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info(f"  Iniciando verificación de modelos")
    log.info(f"  Precisión : {PRECISION.upper()}")
    log.info(f"  Directorio: {MODELS_DIR}")
    log.info("=" * 60)

    validate_env(require_hf_token=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    folders = FOLDERS_BY_PRECISION[PRECISION]
    marker  = get_marker_path()

    # ── Verificación rápida: ¿ya están descargados y son válidos? ──
    if not FORCE_DOWNLOAD and marker.exists():
        log.info("Marker de descarga encontrado. Verificando integridad...")

        if verify_models(folders):
            log.info("Modelos verificados correctamente. Saltando descarga.")
            return
        else:
            log.warning(
                "Marker existe pero la verificación falló "
                "(descarga previa incompleta). Re-descargando..."
            )
            marker.unlink(missing_ok=True)

    if FORCE_DOWNLOAD:
        log.info("FORCE_DOWNLOAD=true detectado. Forzando re-descarga.")
        marker.unlink(missing_ok=True)

    validate_env(require_hf_token=True)

    # ── Descarga ──
    try:
        download_models(folders)
    except Exception as exc:
        log.error(f"Error durante la descarga: {exc}")
        sys.exit(1)

    # ── Verificación post-descarga ──
    log.info("Verificando integridad post-descarga...")

    if not verify_models(folders):
        log.error(
            "La descarga completó pero la verificación de archivos falló. "
            "Revisa el espacio en disco y los permisos."
        )
        sys.exit(1)

    # ── Marcar como listo ──
    marker.touch()
    log.info(f"Modelos {PRECISION.upper()} listos en {MODELS_DIR}")


if __name__ == "__main__":
    main()