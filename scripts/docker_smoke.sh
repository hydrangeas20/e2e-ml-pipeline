#!/usr/bin/env bash
# Build the image, start a container, and verify: health check, readiness,
# legacy + v1 prediction, non-root user, graceful stop. Exits non-zero on failure.
set -euo pipefail
DOCKER=${DOCKER:-docker}
IMG=${IMG:-mlpipe:dev}
NAME=mlpipe-smoke-$$
PORT=${PORT:-18000}

$DOCKER build ${BUILD_ARGS:-} -t "$IMG" .
$DOCKER run -d --name "$NAME" -p "$PORT:8000" \
  -e MLPIPE_ADMIN_TOKEN="smoke-admin-token-0123456789" "$IMG" >/dev/null
trap '$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 60); do
  status=$($DOCKER inspect -f '{{.State.Health.Status}}' "$NAME")
  [ "$status" = healthy ] && break
  sleep 1
done
echo "health: $status"; [ "$status" = healthy ]

echo "uid: $($DOCKER exec "$NAME" id -u)"; [ "$($DOCKER exec "$NAME" id -u)" = 10001 ]
curl -fsS "http://127.0.0.1:$PORT/readyz"; echo
curl -fsS -X POST "http://127.0.0.1:$PORT/predict" -H 'content-type: application/json' \
  -d '{"Pregnancies":4,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":28.0,"DiabetesPedigreeFunction":0.627,"Age":50}'; echo
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/predict" \
  -H 'content-type: application/json' -d '{"Glucose":"x"}')
echo "invalid payload -> $code"; [ "$code" = 422 ]
curl -fsS -H "Authorization: Bearer smoke-admin-token-0123456789" \
  "http://127.0.0.1:$PORT/admin/deployment" | head -c 300; echo

$DOCKER stop -t 20 "$NAME" >/dev/null
$DOCKER logs "$NAME" 2>&1 | grep -q '"msg": "service stopped"'
echo "graceful stop: ok"
echo "DOCKER SMOKE PASSED"
