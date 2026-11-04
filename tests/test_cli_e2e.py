from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlpipe.cli import main
from tests.conftest import DATA, ROOT


def test_cli_offline_lifecycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    reg = str(tmp_path / "reg")
    assert main(["import-legacy", "--registry", reg]) == 0
    assert (
        main(
            [
                "train",
                "--registry",
                reg,
                "--data",
                str(DATA),
                "--version",
                "v2-rf",
                "--params",
                '{"n_estimators": 20}',
            ]
        )
        == 0
    )
    assert main(["make-fixtures", "--registry", reg, "--data", str(DATA)]) == 0
    assert main(["validate", "--registry", reg, "bad-corrupted"]) == 2
    assert main(["validate", "--registry", reg, "v2-rf"]) == 0
    assert main(["bootstrap", "--registry", reg, "v1-legacy-rf"]) == 0
    assert main(["bootstrap", "--registry", reg, "v2-rf"]) == 1  # already bootstrapped
    capsys.readouterr()
    assert main(["state", "--registry", reg]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["active"] == st["known_good"] == "v1-legacy-rf"
    assert st["versions"]["bad-corrupted"]["state"] == "rejected"
    assert main(["events", "--registry", reg, "--limit", "3"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 3
    assert (
        main(
            [
                "diagnose",
                "--registry",
                reg,
                "--version",
                "v1-legacy-rf",
                "--data",
                str(DATA),
                "--simulate-defect",
                "bmi_age_swap",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["likely_root_cause"] == "feature_pipeline"


@pytest.mark.slow
def test_demo_end_to_end_real_server(tmp_path: Path) -> None:
    """The reproducible demonstration (section 6.8) as a test: real uvicorn process."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlpipe.cli",
            "demo",
            "--workdir",
            str(tmp_path / "demo"),
            "--port",
            "8799",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "DEMO PASSED" in out.stdout and "FAIL " not in out.stdout
    rep = json.loads((tmp_path / "demo" / "demo_report.json").read_text())
    assert rep["result"] == "passed"
