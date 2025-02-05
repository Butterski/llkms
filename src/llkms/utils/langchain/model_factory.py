import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI

from llkms.utils.logger import logger


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7


class ModelFactory:
    PROVIDER_CONFIGS = {
        "deepseek": {
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
    }

    @classmethod
    def create_model(cls, config: ModelConfig) -> ChatOpenAI:
        provider_config = cls.PROVIDER_CONFIGS.get(config.provider)
        if not provider_config:
            raise ValueError(f"Unsupported provider: {config.provider}")

        api_key = config.api_key or os.getenv(provider_config["api_key_env"])
        if not api_key:
            raise ValueError(f"API key not found for provider {config.provider}")

        logger.info(f"Creating model {config.model_name} with provider {config.provider}")
        return ChatOpenAI(
            model=config.model_name,
            openai_api_key=api_key,
            openai_api_base=config.api_base or provider_config["api_base"],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    @classmethod
    def get_default_config(cls, provider: str, model_name: str) -> ModelConfig:
        return ModelConfig(provider=provider, model_name=model_name)
