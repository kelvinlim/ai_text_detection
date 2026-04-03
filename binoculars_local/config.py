"""Configuration for local Binoculars service."""

from pydantic_settings import BaseSettings


class BinocularsConfig(BaseSettings):
    model_config = {"env_prefix": "BINOCULARS_"}

    observer_model: str = "tiiuae/falcon-7b"
    performer_model: str = "tiiuae/falcon-7b-instruct"
    device: str = "auto"              # "auto", "mps", "cpu"
    dtype: str = "float16"            # "float16" or "float32"
    threshold_mode: str = "low_fpr"   # "accuracy" or "low_fpr"
    host: str = "0.0.0.0"
    port: int = 8080
    max_chunks_per_request: int = 100


config = BinocularsConfig()
