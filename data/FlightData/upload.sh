#!/bin/bash
export BUCKET=${BUCKET:=nick-flight-data}
echo "Uploading to bucket $BUCKET..."
gsutil -m cp *.csv gs://$BUCKET/flights/raw
