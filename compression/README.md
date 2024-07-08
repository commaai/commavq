# Lossless compression challenge

**Prize: highest compression rate on 5,000 minutes of driving video (~915MB) - Challenge ends July, 1st 2024 11:59pm AOE**

Submit a single zip archive containing
- a compressed version of the first two splits (5,000 minutes) of the commaVQ dataset
- a python script named `decompress.py` to save the decompressed files into their original format in `OUTPUT_DIR`

Everything in this repository and in PyPI is assumed to be available (you can `pip install` in the decompression script), anything else should to be included in the archive.

To evalute your submission, we will run:
```bash
./compression/evaluate.sh path_to_submission.zip
```

| Implementation                                                                     | Compression rate |
| :----------------------------------------------------------------------------------| ---------------: |
| [pkourouklidis](https://github.com/pkourouklidis) (arithmetic coding with GPT)     |  2.6             |
| anonymous (zpaq)                                                                   |  2.3             |
| [rostislav](https://github.com/rostislav) (zpaq)                                   |  2.3             |
| anonymous (zpaq)                                                                   |  2.2             |
| anonymous (zpaq)                                                                   |  2.2             |
| [0x41head](https://github.com/0x41head) (zpaq)                                     |  2.2             |
| [tillinf](https://github.com/tillinf) (zpaq)                                       |  2.2             |
| baseline (lzma)                                                                    |  1.6             |


Have fun!
