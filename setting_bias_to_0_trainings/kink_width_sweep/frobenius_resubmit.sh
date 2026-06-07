#!/bin/bash
# Resubmission of failed/cancelled jobs from lipschitz array job 5515316.
#
# Group 1 — standard memory (40G), covers:
#   _8-10   : guess / uniform / u512,768,1024      (partial COMPLETE, resume)
#   _21     : guess / uniform_02 / u1024           (partial COMPLETE, resume)
#   _30,32  : guess / kaiming_uniform / u512,1024  (partial COMPLETE, resume)
#   _40-42  : guess / kaiming_normal / u256,512,768 (0 COMPLETE, retry)
#   _56     : sgd / uniform_02 / u4               (0 COMPLETE, retry)
#   _75-76  : sgd / kaiming_uniform / u768,1024   (0 COMPLETE, retry)
#   _77-87  : sgd / kaiming_normal / u2-u1024     (0 COMPLETE, start fresh)
#
# Group 2 — reduced target_model_count_subrun (10k, 40G), covers:
#   _33-35  : guess / kaiming_normal / u2,4,8     (OOM at 40G+80G → flush weights every 10k models)

set -e
cd "$(dirname "$0")"

echo "=== Submitting Group 1: standard jobs (40G) ==="
sbatch \
  --job-name=lip_retry \
  --partition=long \
  --cpus-per-task=10 \
  --mem=40G \
  --array=8-10,21,30,32,40-42,56,75-87 \
  --output=logs/lipschitz_%A_%a.out \
  --error=logs/lipschitz_%A_%a.err \
  lipschitz_train.sh

echo "=== Submitting Group 2: guess/kaiming_normal/u2,u4,u8 with reduced target_model_count_subrun ==="
# OOM persists even at MCTBS=96000 (6000 models/batch) because the issue is not
# batch size but the inner while loop running until 500k models accumulate in
# perfect_model_weights before any save. CPU memory allocator cannot reclaim
# pages across hundreds of allocation cycles → monotonically growing RSS.
# Fix: target_model_count_subrun=10000 flushes to disk every 10k models;
# the outer loop repeats until the DB reaches 500k total. Peak in-memory
# weight data stays <1 MB per subrun for these tiny network widths.
sbatch \
  --job-name=lip_kn_small \
  --partition=long \
  --cpus-per-task=10 \
  --mem=40G \
  --array=33-35 \
  --output=logs/lipschitz_%A_%a.out \
  --error=logs/lipschitz_%A_%a.err \
  lipschitz_train_kn_small.sh
