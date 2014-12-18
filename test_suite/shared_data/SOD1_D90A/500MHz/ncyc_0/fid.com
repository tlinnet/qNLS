#!/bin/csh

var2pipe -in ./fid \
 -noaswap  \
  -xN              1600  -yN               256  \
  -xT               800  -yT               128  \
  -xMODE        Complex  -yMODE      Rance-Kay  \
  -xSW        10000.000  -ySW         1400.021  \
  -xOBS         499.862  -yOBS          50.656  \
  -xCAR           4.773  -yCAR         118.344  \
  -xLAB              HN  -yLAB             N15  \
  -ndim               2  -aq2D          States  \
  -out ./test.fid -verb -ov

sleep 5
