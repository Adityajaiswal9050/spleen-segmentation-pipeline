#!/bin/bash
LOGFILE=/home/vboxuser/cv-project/monai-project/nightly_run.log
cd /home/vboxuser/cv-project/monai-project

echo "=== Nightly run: $(date) ===" >> "$LOGFILE"
docker run --rm -v /home/vboxuser/cv-project/monai-project:/app spleen-eval:latest >> "$LOGFILE" 2>&1
python3 scripts/version_and_benchmark.py >> "$LOGFILE" 2>&1
echo "=== Finished: $(date) ===" >> "$LOGFILE"
echo "" >> "$LOGFILE"
