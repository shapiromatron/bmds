from simple_settings import settings
from simple_settings.utils import settings_stub


def test_settings():
    assert settings.BMDS_MODEL_TIMEOUT_SECONDS == 10
    with settings_stub(BMDS_MODEL_TIMEOUT_SECONDS=1):
        assert settings.BMDS_MODEL_TIMEOUT_SECONDS == 1
