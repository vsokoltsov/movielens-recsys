#!/bin/sh
set -e

/entrypoint.sh &

EMAIL="${PGADMIN_DEFAULT_EMAIL:-admin@local}"
USER_DIR="$(echo "$EMAIL" | sed 's/@/_/g; s/\./_/g')"

TARGET_DIR="/var/lib/pgadmin/storage/${USER_DIR}"

echo "Waiting for pgAdmin storage dir: $TARGET_DIR"
for i in $(seq 1 60); do
  if [ -d "$TARGET_DIR" ]; then
    break
  fi
  sleep 1
done

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR: storage dir not created: $TARGET_DIR"
  exit 1
fi

cp /pgpass "${TARGET_DIR}/pgpassfile"
chmod 600 "${TARGET_DIR}/pgpassfile"

wait