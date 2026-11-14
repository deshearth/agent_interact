#!/bin/bash -l
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
/usr/bin/python3 scripts/hpc3_recovery.py tick --cron >> runs/hpc3_recovery_v1/cron.log 2>&1
