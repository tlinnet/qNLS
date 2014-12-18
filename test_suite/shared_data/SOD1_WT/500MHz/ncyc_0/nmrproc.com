#!/bin/csh

#
# Basic 2D Phase-Sensitive Processing:
#   Cosine-Bells are used in both dimensions.
#   Use of "ZF -auto" doubles size, then rounds to power of 2.
#   Use of "FT -auto" chooses correct Transform mode.
#   Imaginaries are deleted with "-di" in each dimension.
#   Phase corrections should be inserted by hand.

nmrPipe -in test.fid \
| nmrPipe -fn POLY -time                               \
| nmrPipe -fn SP -off 0.5 -end 0.98 -pow 2 -c 0.5     \
| nmrPipe  -fn ZF -auto                               \
| nmrPipe  -fn FT -auto                               \
| nmrPipe  -fn PS -p0 79.2 -p1 -79.00 -di -verb         \
| nmrPipe -fn POLY -auto -xn 5.5ppm -ord 1             \
| nmrPipe -fn EXT -sw                                  \
| nmrPipe  -fn TP                                     \
| nmrPipe  -fn SP -off 0.5 -end 0.98 -pow 2 -c 0.5 \
| nmrPipe  -fn ZF -auto                               \
| nmrPipe  -fn FT -neg                               \
| nmrPipe  -fn PS -p0 0.00 -p1 0.00 -di -verb         \
   -ov -out test.ft2
