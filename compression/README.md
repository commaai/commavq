# Lossless compression challenge

You should submit a single zip archive containing
- a compressed version of the first two splits of the commaVQ dataset
- a python script named `decompress.py` to save the decompressed files into their original format in `OUTPUT_DIR`

Everything in this repository and in PyPI is assumed to be available (you can `pip install` in the decompression script).

To evalute your submission, we will run:
```bash
./compression/evaluate.sh path_to_submission.zip
```

| Implementation                                                                     | Compression rate |
| :----------------------------------------------------------------------------------| ---------------: |
| lzma                                                                               |  1.6             |

Have fun!
