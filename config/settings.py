"""Environment-backed settings shared by every application component.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, Mapping


APP_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DISTANCE_MEASURES = {"COSINE", "DOT_PRODUCT", "EUCLIDEAN"}
FIRESTORE_MAX_DIMENSION = 2048
FIRESTORE_MAX_RESULTS = 1000
GEMINI_MAX_OUTPUT_TOKENS = 65_536


class ConfigurationError(ValueError):
    """Raised when application settings are incomplete or invalid."""


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


def _resolve_path(value: str | Path, app_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (app_root / path).resolve()


def _load_project_dotenv() -> None:
    """Load the project-local .env without replacing exported environment variables."""

    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("python-dotenv is required to load the project .env file.") from error
    load_dotenv(dotenv_path=APP_ROOT / ".env", override=False)


def configure_vertex_ai_environment() -> None:
    """Force Google Gen AI and ADK clients to use Vertex AI instead of API keys."""

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings loaded once from the environment."""

    app_root: Path = APP_ROOT
    input_dir: Path = field(default_factory=lambda: APP_ROOT / "input")
    output_dir: Path = field(default_factory=lambda: APP_ROOT / "output")
    insert_into_index: bool = False
    gemini_model: str = "gemini-3.5-flash"
    chat_app_name: str = "agent_ledger_chat"
    chat_max_output_tokens: int = 8000
    chat_temperature: float = 0.2
    chat_top_p: float = 0.9
    chat_search_limit: int = 10
    chat_list_limit: int = 200
    gemini_batch_enabled: bool = True
    gemini_batch_size: int = 128
    gemini_batch_max_bytes: int = 10 * 1024 * 1024
    gemini_batch_poll_seconds: float = 10.0
    gemini_batch_timeout_seconds: int = 24 * 60 * 60
    gemini_batch_fallback: bool = False
    gemini_batch_gcs_uri: str = "gs://test-100-invoices/vertex-batches"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 16
    device: str = "cpu"
    normalize_embeddings: bool = True
    firestore_project_id: str | None = None
    firestore_database: str = "(default)"
    firestore_collection: str = "invoices"
    firestore_vector_field: str = "embedding"
    distance_measure: str = "DOT_PRODUCT"
    google_cloud_location: str = "global"
    max_image_bytes: int = 19 * 1024 * 1024
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Create settings from environment variables with rooted path defaults."""

        if environ is None:
            _load_project_dotenv()
            env = os.environ
        else:
            env = environ
        configure_vertex_ai_environment()
        app_root = _resolve_path(env.get("AGENT_LEDGER_ROOT", APP_ROOT), APP_ROOT)
        return cls(
            app_root=app_root,
            input_dir=_resolve_path(env.get("AGENT_LEDGER_INPUT_DIR", "input"), app_root),
            output_dir=_resolve_path(env.get("AGENT_LEDGER_OUTPUT_DIR", "output"), app_root),
            insert_into_index=_parse_bool(env.get("INSERT_INTO_INDEX", "false")),
            gemini_model=env.get("GEMINI_MODEL", "gemini-3.5-flash"),
            chat_app_name=env.get("CHAT_APP_NAME", "agent_ledger_chat"),
            chat_max_output_tokens=int(env.get("CHAT_MAX_OUTPUT_TOKENS", "8000")),
            chat_temperature=float(env.get("CHAT_TEMPERATURE", "0.2")),
            chat_top_p=float(env.get("CHAT_TOP_P", "0.9")),
            chat_search_limit=int(env.get("CHAT_SEARCH_LIMIT", "10")),
            chat_list_limit=int(env.get("CHAT_LIST_LIMIT", "200")),
            gemini_batch_enabled=_parse_bool(env.get("GEMINI_BATCH_ENABLED", "true")),
            gemini_batch_size=int(env.get("GEMINI_BATCH_SIZE", "128")),
            gemini_batch_max_bytes=int(
                env.get("GEMINI_BATCH_MAX_BYTES", str(10 * 1024 * 1024))
            ),
            gemini_batch_poll_seconds=float(env.get("GEMINI_BATCH_POLL_SECONDS", "10")),
            gemini_batch_timeout_seconds=int(
                env.get("GEMINI_BATCH_TIMEOUT_SECONDS", str(24 * 60 * 60))
            ),
            gemini_batch_fallback=_parse_bool(env.get("GEMINI_BATCH_FALLBACK", "false")),
            gemini_batch_gcs_uri=env.get(
                "GEMINI_BATCH_GCS_URI",
                "gs://test-100-invoices/vertex-batches",
            ).rstrip("/"),
            embedding_model=env.get(
                "EMBEDDING_MODEL",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ),
            embedding_dimension=int(env.get("EMBEDDING_DIMENSION", "384")),
            embedding_batch_size=int(env.get("EMBEDDING_BATCH_SIZE", "16")),
            device=env.get("EMBEDDING_DEVICE", "cpu"),
            normalize_embeddings=_parse_bool(env.get("NORMALIZE_EMBEDDINGS", "true")),
            firestore_project_id=env.get("GOOGLE_CLOUD_PROJECT") or None,
            firestore_database=env.get("FIRESTORE_DATABASE", "(default)"),
            firestore_collection=env.get("FIRESTORE_COLLECTION", "invoices"),
            firestore_vector_field=env.get("FIRESTORE_VECTOR_FIELD", "embedding"),
            distance_measure=env.get("VECTOR_DISTANCE_MEASURE", "DOT_PRODUCT").upper(),
            google_cloud_location=env.get("GOOGLE_CLOUD_LOCATION", "global"),
            max_image_bytes=int(env.get("MAX_IMAGE_BYTES", str(19 * 1024 * 1024))),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
        )

    def validate(
        self,
        *,
        require_cloud: bool = False,
        require_input: bool = True,
    ) -> None:
        """Validate runtime settings and optional cloud or image requirements."""

        if require_input and not self.input_dir.is_dir():
            raise ConfigurationError(f"Image input directory does not exist: {self.input_dir}")
        if self.embedding_dimension <= 0:
            raise ConfigurationError("Embedding dimension must be greater than zero.")
        if self.embedding_dimension > FIRESTORE_MAX_DIMENSION:
            raise ConfigurationError(
                f"Embedding dimension cannot exceed Firestore's {FIRESTORE_MAX_DIMENSION} limit."
            )
        if self.embedding_batch_size <= 0:
            raise ConfigurationError("Embedding batch size must be greater than zero.")
        if not self.chat_app_name.strip():
            raise ConfigurationError("Chat application name cannot be empty.")
        if not 1 <= self.chat_max_output_tokens <= GEMINI_MAX_OUTPUT_TOKENS:
            raise ConfigurationError(
                f"Chat output token limit must be between 1 and {GEMINI_MAX_OUTPUT_TOKENS}."
            )
        if not 0.0 <= self.chat_temperature <= 2.0:
            raise ConfigurationError("Chat temperature must be between 0.0 and 2.0.")
        if not 0.0 < self.chat_top_p <= 1.0:
            raise ConfigurationError("Chat top-p must be greater than 0.0 and at most 1.0.")
        if not 1 <= self.chat_search_limit <= FIRESTORE_MAX_RESULTS:
            raise ConfigurationError(
                f"Chat search limit must be between 1 and {FIRESTORE_MAX_RESULTS}."
            )
        if not 1 <= self.chat_list_limit <= FIRESTORE_MAX_RESULTS:
            raise ConfigurationError(
                f"Chat list limit must be between 1 and {FIRESTORE_MAX_RESULTS}."
            )
        if self.gemini_batch_size <= 0:
            raise ConfigurationError("Gemini batch size must be greater than zero.")
        if not 0 < self.gemini_batch_max_bytes <= 10 * 1024 * 1024:
            raise ConfigurationError("Gemini image batches cannot exceed the 10 MB safety cap.")
        if self.gemini_batch_poll_seconds <= 0:
            raise ConfigurationError("Gemini batch polling interval must be greater than zero.")
        if self.gemini_batch_timeout_seconds <= 0:
            raise ConfigurationError("Gemini batch timeout must be greater than zero.")
        if self.gemini_batch_enabled and not self.gemini_batch_gcs_uri.startswith("gs://"):
            raise ConfigurationError("GEMINI_BATCH_GCS_URI must start with gs://.")
        if self.max_image_bytes <= 0:
            raise ConfigurationError("Maximum image size must be greater than zero.")
        if self.distance_measure not in SUPPORTED_DISTANCE_MEASURES:
            raise ConfigurationError(
                "Vector distance measure must be COSINE, DOT_PRODUCT, or EUCLIDEAN."
            )
        if self.distance_measure == "DOT_PRODUCT" and not self.normalize_embeddings:
            raise ConfigurationError(
                "DOT_PRODUCT requires unit-normalized embeddings; enable normalization "
                "or use COSINE."
            )
        if not self.firestore_collection.strip():
            raise ConfigurationError("Firestore collection cannot be empty.")
        if not self.firestore_vector_field.strip():
            raise ConfigurationError("Firestore vector field cannot be empty.")
        if require_cloud and not self.firestore_project_id:
            raise ConfigurationError("GOOGLE_CLOUD_PROJECT is required for Firestore operations.")
        if require_cloud and not self.google_cloud_location:
            raise ConfigurationError("GOOGLE_CLOUD_LOCATION is required for Vertex AI.")

    def prepare_output(self) -> None:
        """Create writable runtime output directories."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "extracted").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_logger() -> Any:
    """Return one shared logger, preferring Loguru when it is installed."""

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        from loguru import logger

        return logger.bind(application="agent-ledger")
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return logging.getLogger("agent-ledger")
