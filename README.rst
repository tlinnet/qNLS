====
qNLS 
====

Asses quality of NLS for coMDD processing.

A python script to quickly evaluate effect of co-processing several 2D spectra together.

The analysis can be performed in 10-15 minutes time, and is a valuable guide to determine which limit one should set the sampling fraction level to.

This script is for the intended use, where several experiments is recorded together, and where the modulation of the peaks are expected only to be in the peak intensities.
That means, that all spectra should share the same frequency information, but not same amplitude.

================
Data requirement
================
First record one full experiment, for example the reference spectrum.

Process the spectrum with NMRPipe, to produce:
fid (original file)
fid.com which produces test.fid
nmrproc.com which produces test.ft2 from test.fid

The script expects that names of files in the directory are:
fid
fid.com
test.fid
nmrproc.com

=================
Outline of script
=================
Put a link to qNLS.py somewhere in your PATH

In the folder, write:
qNLS.py -sf 20 15 10 5

Where sf is the sampling fractions in percent.

The script then proceeds as follows:

* Loop over the sampling fraction level: 
  * Produce nls.in
    * Numper of points in direct and in-direct dimensions, and sweep-width (Hz) are read from test.fid with nmrglue.
    * In 00_ref.fid, write nls.in at 100 sf level
    * In 01.fid, write nls.in according to the sampling fraction level
  * Produce nusschedules with 'nussampler' from MddNMR within each *.fid folder
  * Binary read the orginal fid file, and write out a truncated fid file in each *.fid folder according to the nusschedule.
    * The individual fid file only contains the in-direct FIDs determined and ordered from the nusscedule.
    * The 00_ref.fid then contains the full fid, but where the in-direct FIDs have been shuffled according to the full nusscedule.
  * Then write proc.sh, which is command file with environment settings to MddNMR. These settings can be modified by parsing input commands to qNLS.py
  * Then write fidSP.com and recFT.com. These are spectrum processing parameters, and these are read from the initial files of fid.com, and nmrproc.com.
  * Then produce files for coMDD processing.
  * Then coMDD process.
  * Then analyse and produce figures.

==================
Software depencies
==================

python module depencies for the script
--------------------------------------
| **nmrglue**, to read spectrum as numpy array  
| **matplotlib.pyplot**, to produce figures  
| **scipy.optimize**, to fit histogram  
| **numpy'', for data arrays  

MddNMR
------


======================
Citations for software
======================

MddNMR
-------
| Orekhov, V.Y. and V.A. Jaravine  
| Analysis of non-uniformly sampled spectra with Multi-?Dimensional Decomposition.  
| Prog. Nucl. Magn. Reson. Spectrosc., 2011
| doi:10.1016/j.pnmrs.2011.02.002 
|  
| Kazimierczuk, K. and V.Y. Orekhov
| Accelerated NMR Spectroscopy by Using Compressed Sensing.  
| Angew. Chem.-Int. Edit., 2011, 123, 5670-3  
|  
| Download & Manual: http://pc8.nmr.gu.se/~mdd/Downloads/  
| Link to discussion: https://groups.google.com/forum/#!forum/mddnmr  

nmrglue
-------
| J.J. Helmus, C.P. Jaroniec  
| Nmrglue: An open source Python package for the analysis of multidimensional NMR data | 
| J. Biomol. NMR 2013, 55, 355-367
| http://dx.doi.org/10.1007/s10858-013-9718-x
|  
| Homepage: http://www.nmrglue.com/  
| Link to discussion: https://groups.google.com/forum/#!forum/nmrglue-discuss  
| The code is develop at Github: https://github.com/jjhelmus/nmrglue/  
| Documentation: http://nmrglue.readthedocs.org/en/latest/index.html  


================
Trouble shooting
================

coMDD in MddNMR needs some standard packages to be present on system.

| On redhat 6, these are the packages to install.   
| > yum compat-libf2c-34  
| > yum install glibc.i686  



