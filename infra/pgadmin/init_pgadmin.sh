#!/bin/sh
set -e

# запускаем стандартный entrypoint в фоне, чтобы pgAdmin успел создать storage
/entrypoint.sh &

# ждём появления storage пользователя (имя папки делается из email)
# pgAdmin обычно заменяет @ и . на _
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

# кладём pgpass в storage и ставим правильные права
cp /pgpass "${TARGET_DIR}/pgpassfile"
chmod 600 "${TARGET_DIR}/pgpassfile"

# ждём основной процесс
wait