#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_submission.zip>"
  exit 1
fi

ZIP_FILE="$1"
DECOMPRESSED_DIR="./compression_challenge_submission_decompressed/"

rm -rf $DECOMPRESSED_DIR
unzip $ZIP_FILE -d $DECOMPRESSED_DIR
OUTPUT_DIR=$DECOMPRESSED_DIR python ./compression_challenge_submission_decompressed/decompress.py
UNPACKED_ARCHIVE=$DECOMPRESSED_DIR PACKED_ARCHIVE=$ZIP_FILE python ./compression/evaluate.py
