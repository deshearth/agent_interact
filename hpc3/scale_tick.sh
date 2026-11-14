#!/bin/bash -l
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
/usr/bin/python3 scripts/hpc3_scale_controller.py tick --cron >> runs/hpc3_scale_3200_v1/cron.log 2>&1
