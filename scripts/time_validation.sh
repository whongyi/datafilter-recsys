#!/bin/sh

#validate on 1 year complete user records (baseline)
for l in 0.1 0.01 0.001 0.0001; do
   python3 "time_filter_exp/time_validation.py" "movielens" $1 $l  "2015-01-01" 2014_1
done  

#validate on different settings (P,N)
for P in 1.0 0.75 0.5 0.25; do 
   for N in 1 7 14 30 60 90 180; do
       for l in 0.1 0.01 0.001 0.0001; do
           python3 "time_filter_exp/time_validation.py" "movielens" $1 $l "2015-01-01" 2014_1 $P $N
       done
   done
done

# validate on complete user records for time intervals of varying length
for year in 2013 2012 2011 2010; do
    for month in 7 1; do
        for l in 0.1 0.01 0.001 0.0001; do
            python3 "time_filter_exp/time_validation.py" "movielens" $1 $l "2015-01-01" $year"_"$month 
        done
    done
done

