#!/bin/csh

var2pipe -in ./fid \
 -noaswap  \
  -xN              1600  -yN               256  \
  -xT               800  -yT               128  \
  -xMODE        Complex  -yMODE      Rance-Kay  \
  -xSW        10000.000  -ySW         2550.045  \
  -xOBS         750.217  -yOBS          76.027  \
  -xCAR           4.677  -yCAR         116.577  \
  -xLAB              HN  -yLAB             N15  \
  -ndim               2  -aq2D          States  \
  -out ./test.fid -verb -ov

sleep 5
