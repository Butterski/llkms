import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from llkms.utils.langchain.model_factory import ModelConfig, ModelFactory


@pytest.fixture
def mock_env_keys(monkeypatch):
    """Set up mock API keys in environment"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-key")


def assert_secret_str_value(secret: SecretStr, expected: str):
    """Helper function to compare SecretStr values"""
    if not isinstance(secret, SecretStr):
        return False
    return secret.get_secret_value() == expected


@pytest.mark.parametrize(
    "config_data,expected",
    [
        (
            {"provider": "deepseek", "model_name": "deepseek-chat", "max_tokens": 1024, "temperature": 0.7},
            {"api_base": "https://api.deepseek.com", "api_key": "mock-deepseek-key"},
        ),
        (
            {"provider": "openai", "model_name": "gpt-3.5-turbo", "max_tokens": 2048, "temperature": 0.5},
            {"api_base": "https://api.openai.com/v1", "api_key": "mock-openai-key"},
        ),
        (
            {
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "custom-key",
                "api_base": "https://custom-api.com",
            },
            {"api_base": "https://custom-api.com", "api_key": "custom-key"},
        ),
    ],
)
def test_model_creation(mock_env_keys, config_data, expected):
    """Test model creation with different configurations"""
    config = ModelConfig(**config_data)
    model = ModelFactory.create_model(config)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == config.model_name
    assert model.openai_api_base == expected["api_base"]
    assert assert_secret_str_value(model.openai_api_key, expected["api_key"])


@pytest.mark.parametrize(
    "provider,error_message",
    [("unsupported", "Unsupported provider: unsupported"), ("deepseek", "API key not found for provider deepseek")],
)
def test_model_creation_errors(provider, error_message, monkeypatch):
    """Test error cases in model creation"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ModelConfig(provider=provider, model_name="test-model")
    with pytest.raises(ValueError) as exc_info:
        ModelFactory.create_model(config)
    assert error_message in str(exc_info.value)
