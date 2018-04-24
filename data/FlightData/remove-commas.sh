#/bin/bash

for month in `seq -w 1 12`; do
    sed 's/,$//g' 2017$month.csv | sed 's/"//g' > tmp
    mv tmp 2017$month.csv
done

