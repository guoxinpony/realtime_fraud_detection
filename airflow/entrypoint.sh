#!/usr/bin/env bash
set -euo pipefail

# Default variables (can be overridden by docker-compose's environment)
: "${_AIRFLOW_WWW_USER_USERNAME:=admin}"
: "${_AIRFLOW_WWW_USER_PASSWORD:=admin}"
: "${_AIRFLOW_WWW_USER_FIRSTNAME:=Air}"
: "${_AIRFLOW_WWW_USER_LASTNAME:=Flow}"
: "${_AIRFLOW_WWW_USER_EMAIL:=admin@example.com}"
: "${AIRFLOW_HOME:=/opt/airflow}"

# Idempotent initialization: Use a file marker. 
# It is recommended to mount AIRFLOW_HOME on 
  # a volume so that multiple restarts will not cause repeated initialization.
if [ ! -f "${AIRFLOW_HOME}/.initialized" ]; then
  echo "[bootstrap] Initializing Airflow DB..."
  airflow db init || true

  echo "[bootstrap] Ensuring admin user exists..."
  # The output of users list is unstable. 
  # Here we use grep -w to match the username exactly.
  if ! airflow users list | grep -wq "${_AIRFLOW_WWW_USER_USERNAME}"; then
    airflow users create \
      --username "${_AIRFLOW_WWW_USER_USERNAME}" \
      --password "${_AIRFLOW_WWW_USER_PASSWORD}" \
      --firstname "${_AIRFLOW_WWW_USER_FIRSTNAME}" \
      --lastname "${_AIRFLOW_WWW_USER_LASTNAME}" \
      --role Admin \
      --email "${_AIRFLOW_WWW_USER_EMAIL}"
  fi

  touch "${AIRFLOW_HOME}/.initialized"
fi

if [ "$#" -eq 0 ]; then
  echo "[bootstrap] Done. No command passed; exiting 0 for airflow-init."
  exit 0
fi

# Return to the official entrypoint that comes with the image 
  # (which will correctly handle commands such as webserver/scheduler/celery)
exec /entrypoint "$@"