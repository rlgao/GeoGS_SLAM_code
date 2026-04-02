#!/bin/bash

bash eval/run_replica.sh  # check: base.yaml
bash eval/run_tum.sh
bash eval/run_waymo.sh
