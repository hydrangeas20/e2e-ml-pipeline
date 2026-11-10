"""Entry point kept at its original path so `uvicorn app.main:app` keeps working.

The implementation lives in ``mlpipe.serving.app``. The original endpoints are
preserved: ``GET /`` returns ``{"status": "ok"}`` and ``POST /predict`` accepts the
same flat feature payload and still returns ``prediction`` and ``confidence``
(now with strict validation and model-version metadata added).
"""

from mlpipe.serving.app import create_app

app = create_app()
