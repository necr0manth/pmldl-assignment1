import sys
from pathlib import Path

import pytest
from pmldl.service import install, render, unit_quote


def test_unit_paths_are_rendered_and_quoted():
    result = render(Path('/tmp/project with spaces%'), Path('/opt/venv/bin/python'))
    assert 'WorkingDirectory="/tmp/project with spaces%%"' in result
    assert 'ExecStart="/opt/venv/bin/python" "/tmp/project with spaces%%/pipeline.py"' in result
    assert '@WORKDIR@' not in result and '@PYTHON@' not in result
    assert 'schedule --interval 300' in result and 'Restart=on-failure' in result


def test_unit_installer_uses_actual_interpreter(tmp_path):
    (tmp_path / 'pipeline.py').write_text('')
    path = install(tmp_path, Path(sys.executable), tmp_path / 'units', 'demo.service')
    assert path.is_file() and sys.executable in path.read_text()


@pytest.mark.parametrize('path', ['bad\npath', 'bad\rpath', 'bad\x00path'])
def test_unit_injection_is_rejected(path):
    with pytest.raises(ValueError):
        unit_quote(path)


def test_bad_unit_name_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        install(tmp_path, Path(sys.executable), tmp_path, '../bad.service')
