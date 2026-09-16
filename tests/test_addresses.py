import pytest
from pmldl import deployment


@pytest.mark.parametrize("host,expected", [("127.0.0.1", "127.0.0.1"), ("0.0.0.0", "127.0.0.1"),
                                           ("::", "[::1]"), ("2001:db8::1", "[2001:db8::1]")])
def test_custom_service_addresses(monkeypatch, host, expected):
    monkeypatch.setenv("BIND_ADDRESS", host)
    monkeypatch.setenv("API_PORT", "18000")
    assert deployment.service_url("API_PORT", "8000") == f"http://{expected}:18000"
