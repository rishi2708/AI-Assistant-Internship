import config as config_module
from config import AppConfig, ensure_directories


def test_config_from_env_has_defaults(monkeypatch):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = AppConfig.from_env()

    assert config.gemini_text_model
    assert not config.has_gemini_key


def test_ensure_directories_runs():
    ensure_directories()
