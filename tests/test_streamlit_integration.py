from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

pytest.importorskip("streamlit", reason="Install requirements-dev.txt for the Streamlit integration test")
from streamlit.testing.v1 import AppTest

from pmldl.api import create_app
from pmldl.common import ROOT


def test_form_gets_prediction_from_api(artifact, monkeypatch):
    with TestClient(create_app(artifact)) as client:
        post = Mock(side_effect=lambda url, json, timeout: client.post("/predict", json=json))
        monkeypatch.setattr(requests, "post", post)
        monkeypatch.setattr(requests, "get", lambda url, timeout: client.get("/model"))
        app = AppTest.from_file(str(ROOT / "code/pmldl/webapp.py")).run()
        assert len(app.number_input) == 4 and len(app.button) == 1
        assert not app.exception
        app.button[0].click().run()
        assert post.call_count == 1
        assert not app.exception
        assert "setosa" in app.success[0].value
        assert "test-run" in app.caption[0].value


def test_form_reports_api_failure(monkeypatch):
    failure = Mock(side_effect=requests.ConnectionError("API offline"))
    monkeypatch.setattr(requests, "post", failure)
    monkeypatch.setattr(requests, "get", failure)
    app = AppTest.from_file(str(ROOT / "code/pmldl/webapp.py")).run()
    app.button[0].click().run()
    assert not app.exception
    assert "Prediction failed" in app.error[0].value
    assert len(app.success) == 0
