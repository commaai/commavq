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
<!-- TABLE-START -->
<table class="ranked">
 <thead>
  <tr>
   <th>
   </th>
   <th>
    score
   </th>
   <th>
    name
   </th>
   <th>
    method
   </th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>
   </td>
   <td>
    3.4
   </td>
   <td>
    <a href="https://github.com/szabolcs-cs">
     szabolcs-cs
    </a>
   </td>
   <td>
    self-compressing neural network
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.9
   </td>
   <td>
    <a href="https://github.com/BradyWynn">
     BradyWynn
    </a>
   </td>
   <td>
    arithmetic coding with GPT
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.6
   </td>
   <td>
    <a href="https://github.com/pkourouklidis">
     pkourouklidis
    </a>
    👑
   </td>
   <td>
    arithmetic coding with GPT
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.3
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.3
   </td>
   <td>
    <a href="https://github.com/rostislav">
     rostislav
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    anonymous
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/0x41head">
     0x41head
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/tillinf">
     tillinf
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/ylevental">
     ylevental
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    2.2
   </td>
   <td>
    <a href="https://github.com/nuniesmith">
     nuniesmith
    </a>
   </td>
   <td>
    zpaq
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td>
    1.6
   </td>
   <td>
    baseline
   </td>
   <td>
    lzma
   </td>
  </tr>
 </tbody>
</table>
<!-- TABLE-END -->

Have fun!
