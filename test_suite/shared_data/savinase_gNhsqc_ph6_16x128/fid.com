#!/bin/csh

var2pipe -in ./fid \
 -noaswap  \
  -xN              2048  -yN               256  \
  -xT              1024  -yT               128  \
  -xMODE        Complex  -yMODE      Rance-Kay  \
  -xSW        12001.200  -ySW         1945.000  \
  -xOBS         750.061  -yOBS          76.012  \
  -xCAR           4.773  -yCAR         119.475  \
  -xLAB              HN  -yLAB             N15  \
  -ndim               2  -aq2D          States  \
  -out ./test.fid -verb -ov

sleep 5
