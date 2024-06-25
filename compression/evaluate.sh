#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_submission.zip>"
  exit 1
fi

ZIP_FILE="$(realpath $1)"
DECOMPRESSED_DIR="./compression_challenge_submission_decompressed/"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR

rm -rf $DECOMPRESSED_DIR
unzip $ZIP_FILE -d $DECOMPRESSED_DIR
OUTPUT_DIR=$DECOMPRESSED_DIR python ./compression_challenge_submission_decompressed/decompress.py
UNPACKED_ARCHIVE=$DECOMPRESSED_DIR PACKED_ARCHIVE=$ZIP_FILE python $DIR/evaluate.py
