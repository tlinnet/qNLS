#!/bin/csh

var2pipe -in ./fid \
 -noaswap  \
  -xN              1024  -yN               256  \
  -xT               512  -yT               128  \
  -xMODE        Complex  -yMODE      Rance-Kay  \
  -xSW         8000.000  -ySW         1800.018  \
  -xOBS         599.891  -yOBS          60.793  \
  -xCAR           4.773  -yCAR         118.005  \
  -xLAB              HN  -yLAB             N15  \
  -ndim               2  -aq2D          States  \
  -out ./test.fid -verb -ov

sleep 5
