"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model paths
    model_cache_dir: Path = Field(default=Path("./models"), description="Model cache directory")
    hf_home: Path = Field(default=Path("./models/.cache"), description="Hugging Face cache")

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    ws_max_size: int = Field(default=10485760, description="WebSocket max message size (10MB)")

    # Processing configuration
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    audio_chunk_size: float = Field(default=2.0, description="Audio chunk size in seconds")
    vad_threshold: float = Field(default=0.5, description="VAD threshold (0-1)")
    buffer_size: float = Field(default=10.0, description="Audio buffer size in seconds")

    # Thresholds (for Phase 2+)
    similarity_threshold_low: float = Field(default=0.60, description="Low similarity threshold")
    similarity_threshold_high: float = Field(
        default=0.75, description="High similarity threshold"
    )
    speaker_verification_threshold: float = Field(
        default=0.75, description="Speaker verification threshold"
    )
    cheating_threshold: float = Field(default=10.0, description="Cheating detection threshold")
    decay_factor: float = Field(default=0.9, description="Suspicion score decay factor")

    # Storage
    storage_root: Path = Field(default=Path("./storage"), description="Storage root directory")
    debug_save_audio: bool = Field(default=False, description="Save audio for debugging")

    # Device
    torch_device: Literal["cuda", "cpu"] = Field(default="cpu", description="PyTorch device (cuda/cpu)")

    # Model names
    vad_model_name: str = Field(default="silero_vad", description="VAD model name")
    stt_model_name: str = Field(
        default="vinai/PhoWhisper-small", description="STT model name"
    )
    stt_model_override: Optional[Path] = Field(
        default=None,
        description="Optional: Path to converted STT model (overrides stt_model_name)",
        alias="STT_MODEL_PATH"
    )

    # SLM configuration (Phase 3)
    slm_enabled: bool = Field(default=True, description="Enable SLM reasoning layer")
    slm_model_path: Optional[Path] = Field(
        default=None,
        description="Path to GGUF model file (e.g. models/slm/qwen2.5-3b-instruct-q4_k_m.gguf)",
        alias="SLM_MODEL_PATH"
    )
    slm_n_gpu_layers: int = Field(
        default=0,
        description="Number of layers to offload to GPU (0=CPU only, -1=all)",
        alias="SLM_N_GPU_LAYERS"
    )
    slm_max_tokens: int = Field(default=4, description="Max tokens for SLM output (YES/NO)")
    slm_context_length: int = Field(default=512, description="SLM context window size")

    @property
    def slm_model_dir(self) -> Path:
        """Get SLM model directory."""
        return self.model_cache_dir / "slm"

    @property
    def vad_model_path(self) -> Path:
        """Get VAD model path."""
        return self.model_cache_dir / "vad"

    @property
    def stt_model_path(self) -> Path:
        """Get STT model path."""
        return self.model_cache_dir / "stt"

    @property
    def enrollment_dir(self) -> Path:
        """Get enrollment directory."""
        return self.storage_root / "enrollment"

    @property
    def transcripts_dir(self) -> Path:
        """Get transcripts directory."""
        return self.storage_root / "transcripts"

    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.storage_root / "logs"

    @property
    def reports_dir(self) -> Path:
        """Get reports directory."""
        return self.storage_root / "reports"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.model_cache_dir,
            self.hf_home,
            self.storage_root,
            self.enrollment_dir,
            self.transcripts_dir,
            self.logs_dir,
            self.reports_dir,
            self.vad_model_path,
            self.stt_model_path,
            self.slm_model_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
