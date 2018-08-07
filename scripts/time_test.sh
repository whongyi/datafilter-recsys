#!/bin/sh
# trap "exit" INT

config_dir="configs/"

while read config; do
    while read date; do
        python3 "time_filter_exp/time_test.py" movielens $config $date $2
    done < ${config_dir}"test_dates.txt"
done < ${config_dir}$1"_"$2"_test_config.txt"

# example: ./scripts/time_test.sh filter pmf Recall 
