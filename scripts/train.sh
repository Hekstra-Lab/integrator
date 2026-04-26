#!/bin/bash
cfg='/Users/luis/temp/data/test_yaml.yaml'

mkdir dir_test_run/
cp $cfg dir_test_run/copy_config.yaml

integrator.train \
  --config /Users/luis/temp/data/test_yaml.yaml \
  --run-dir dir_test_run

integrator.pred \
  --run-dir dir_test_run \
  --write-refl

