#! /usr/bin/env python

###############################################################################
#                                                                             #
# Copyright (C) 2014-2015 Troels E. Linnet, SBiNLab, Copenhagen University    #
# Copyright (C) 2014-2015 Kaare Teilum, SBiNLab, Copenhagen University        #
#                                                                             #
# This file is part of the program relax (http://www.nmr-relax.com).          #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################

import csv
import glob
import nmrglue
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy
from numpy.ma import masked_where
import os
import os.path
import random
import shutil
import subprocess
from scipy.optimize import leastsq
from stat import S_IRWXU, S_IRGRP, S_IROTH
import sys
import time
from warnings import warn

# Get start time
start_time = time.time()

# Add arguments to the script.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', nargs='?', type=int, metavar='N', dest='N', default=1, help='Number of repeated spectra. -N 1')
#parser.add_argument('-sf', nargs='+', type=int, metavar='sf', dest='sf', default=[60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8], help='list of sampling fractions in percent: -sf 24 20 16 12 8')
#parser.add_argument('-sf', nargs='+', type=int, metavar='sf', dest='sf', default=[40, 28, 16, 12, 8], help='list of sampling fractions in percent: -sf 24 20 16 12 8')
parser.add_argument('-sf', nargs='+', type=int, metavar='sf', dest='sf', default=[80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8], help='list of sampling fractions in percent: -sf 24 20 16 12 8')
parser.add_argument('-T2', nargs='?', type=float, metavar='T2', dest='T2', default=0.1, help='T2: spin-spin relaxation time, the expected time constant characterizing the signal decay in in-direct dimension (s): -T2 0.1.')
parser.add_argument('-FST_PNT_PPM', nargs='?', type=float, metavar='FST_PNT_PPM', dest='FST_PNT_PPM', default=11, help='MDD param: Define from where the region of interest starts in the direct dimension [ppm]: -FST_PNT_PPM 11')
parser.add_argument('-ROISW', nargs='?', type=float, metavar='ROISW', dest='ROISW', default=5, help='MDD param: Define sweep-width in ppm, to subtract from FST_PNT_PPM: -ROISW 5')
parser.add_argument('-SRSIZE', nargs='?', type=float, metavar='SRSIZE', dest='SRSIZE', default=0.1, help='MDD param: Size of sub-region (ppm): -SRSIZE 0.1')
parser.add_argument('-CEXP', nargs='?', type=str, metavar='CEXP', dest='CEXP', default='yn', help='Toggle R-MDD / MDD mode. For a dimension, with "y" time domain shape in the dimension is expected to be autoregressive. In other words, we assume that the FID in the dimension is a complex exponent. CEXP=y may be used, for example, for HNCO and HNcoCA experiments, but not for the NOESYs: -CEXP yn')
parser.add_argument('-MDDTHREADS', nargs='?', type=int, metavar='MDDTHREADS', dest='MDDTHREADS', default=16, help='MDD param: Maximal number of parallel processes: -MDDTHREADS 16')
parser.add_argument('-NCOMP', nargs='?', type=int, metavar='NCOMP', dest='NCOMP', default=25, help='MDD param: Number of components per sub-region: -NCOMP 25')
parser.add_argument('-NITER', nargs='?', type=int, metavar='NITER', dest='NITER', default=50, help='MDD param: number of iteration in mddnmr: -NITER 50')
parser.add_argument('-MDD_NOISE', nargs='?', type=float, metavar='MDD_NOISE', dest='MDD_NOISE', default=0.7, help='MDD param: Noise in mddnmr: -MDD_NOISE 0.7')
parser.add_argument('-N_NUS_SCHEDULES', nargs='?', type=int, metavar='N_NUS_SCHEDULES', dest='N_NUS_SCHEDULES', default=10, help='dpoisson7.1 param: Number of NUS scedules to produce for scoring: -N_NUS_SCHEDULES 1000')
input_args = parser.parse_args()

def call_prog(args=None, verbose=True):
    """Call an external program.

    @keyword args:    List of arguments to call
    @type args:       list of str
    @keyword verbose: A flag which if True, will print to screen.
    @type verbose:    bool
    @return:          The return code from the program, and the list of lines from output.
    @rtype:           int, list of str
    """

    # Call function.
    Temp = subprocess.Popen(args, stdout=subprocess.PIPE)
     
    ## But do not wait until program finish, start displaying output immediately
    if verbose:
        while True:
            out = Temp.stdout.read(1)
            if out == '' and Temp.poll() != None:
                break
            if out != '':
                sys.stdout.write(out)
                sys.stdout.flush()

    # Communicate with program, and get output and error output.
    (output, errput) = Temp.communicate()

    # Wait for finish and get return code.
    returncode = Temp.wait()

    # Split the output into lines.
    line_split = output.splitlines()

    if returncode == 1:
        print("\nProgram call returned error code: %s\n"%args[0])
        raise Exception("Program call returned error code %s."%returncode)

    return returncode, line_split


def check_files(files=None):
    """Check list of files is available in current directory.

    @keyword files:     List of expected filenames in current directory
    @type files:        list of str
    """

    # Loop over all files.
    for fname in files:
        test = os.path.isfile(fname)
        if not test:
            print("\nMissing a file. All these files are expected to be present: %s\n"%files)
            raise Exception("The file '%s' was not found in directory."%fname)


def func_gauss(params=None, x=None):
    """Calculate the Gaussian distribution for a given x value.

    @param params:  The vector of parameter values.
    @type params:   numpy rank-1 float array
    @keyword x:     The x value to calculate the probability for.
    @type x:        numpy array
    @return:        The probability corresponding to x.
    @rtype:         float
    """

    # Unpack,
    # a: The amplitude of the distribution.
    # mu: The center of the distribution.
    # sigma: The standard deviation of the distribution.
    a, mu, sigma = params

    # Calculate and return the probability.
    return a*numpy.exp(-(x-mu)**2/(2*sigma**2))


def func_gauss_residual(params=None, x=None, values=None):
    """Calculate the residual vector betwen measured values and the function values.

    @param params:  The vector of parameter values.
    @type params:   numpy rank-1 float array
    @keyword x:     The x data points.
    @type x:        numpy array
    @param values:  The measured values.
    @type values:   numpy array
    @return:        The residuals.
    @rtype:         numpy array
    """

    # Let the vector K be the vector of the residuals. A residual is the difference between the observation and the equation calculated using the initial values.
    K = values - func_gauss(params=params, x=x)

    # Return
    return K


def func_gauss_weighted_residual(params=None, x=None, values=None, errors=None):
    """Calculate the weighted residual vector betwen measured values and the function values.

    @param params:  The vector of parameter values.
    @type params:   numpy rank-1 float array
    @keyword x:     The x data points.
    @type x:        numpy array
    @param values:  The measured values.
    @type values:   numpy array
    @param errors:  The standard deviation of the measured intensity values per time point.
    @type errors:   numpy array
    @return:        The weighted residuals.
    @rtype:         numpy array
    """

    # Let the vector Kw be the vector of the weighted residuals. A residual is the difference between the observation and the equation calculated using the initial values.
    Kw = 1. / errors * func_gauss_residual(params=params, x=x, values=values)

    # Return
    return Kw


def hist_plot(ndarray=None, hist_kwargs=None, show=False):
    """Flatten the 2D numpy array, and plot as histogram.

    @keyword ndarray:           The numpy array to flatten, and plot as histogram.
    @type ndarray:              numpy array
    @keyword hist_kwargs:       The dictionary of keyword arguments to be send to matplotlib.pyplot.hist() plot function.  If None, standard values will be used.
    @type hist_kwargs:          None or dic
    @keyword show:              A flag which if True will make a call to matplotlib.pyplot.show().
    @type show:                 bool
    @return:                    The matplotlib.axes.AxesSubplot class, which can be manipulated to add additional text to the axis.
    @rtype:                     matplotlib.axes.AxesSubplot
    """

    # Flatten the numpy data array.
    data = ndarray.flatten()

    # Now make a histogram.
    # http://matplotlib.org/1.2.1/examples/api/histogram_demo.html
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if hist_kwargs == None:
        hist_kwargs = {'bins': 3000, 'range': None, 'normed': False, 'facecolor':'green', 'alpha':0.75}

    # Make the plot, and unpack the dictionary keywords.
    #n : array or list of arrays. The values of the histogram bins.
    #bins : array. The edges of the bins.
    #patches : list or list of lists. Silent list of individual patches used to create the histogram.
    n, bins, patches = ax.hist(data, **hist_kwargs)

    # Calculate the bin centers.
    bincenters = 0.5*(bins[1:]+bins[:-1])

    # Find index for maximum number in a bin.
    i = numpy.argmax(n)

    # Get the position for the maximum.
    bin_max_x = bincenters[i]

    # Get the amplitude for the maximum.
    bin_max_y = n[i]

    # Try find Full width at half maximum (FWHM). FWHM = 2 * sqrt(2 ln(2 )) * sigma ~ 2.355 * sigma.
    # Half maximum
    hm = 0.5 * bin_max_y

    # Find the first instances of left and right bin, where is lower than hm.
    for j in range(1, len(bins)):
        # Find the center values of the bins.
        left_bin_x = bincenters[i-j]
        right_bin_x = bincenters[i+j]

        # Find the values of the bins.
        left_bin_y = n[i-j]
        right_bin_y = n[i+j]

        if left_bin_y < hm and right_bin_y < hm:
            fwhm = right_bin_x - left_bin_x
            fwhm_std = fwhm / (2. * numpy.sqrt(2. * numpy.log(2.)))
            break

    # Define function to minimise.
    t_func = func_gauss_weighted_residual

    # All args to function. Params are packed out through function, then other parameters.
    # N is number of observations.
    N = len(bincenters)

    errors = numpy.ones(N)
    args=(bincenters, n, errors)

    # Initial guess for minimisation.
    x0 = numpy.asarray( [bin_max_y, bin_max_x, fwhm_std] )

    # Call scipy.optimize.leastsq.
    #ftol:              The function tolerance for the relative error desired in the sum of squares, parsed to leastsq.
    #xtol:              The error tolerance for the relative error desired in the approximate solution, parsed to leastsq.
    #maxfev:            The maximum number of function evaluations, parsed to leastsq.  If zero, then 100*(N+1) is the maximum function calls.  N is the number of elements in x0=[r2eff, i0].
    #factor:            The initial step bound, parsed to leastsq.  It determines the initial step bound (''factor * || diag * x||'').  Should be in the interval (0.1, 100).
    popt, pcov, infodict, errmsg, ier = leastsq(func=t_func, x0=x0, args=args, full_output=True, ftol=1e-15, xtol=1e-15, maxfev=10000000, factor=100.0)

    # Catch errors:
    if ier in [1, 2, 3, 4]:
        warning = None
    elif ier in [9]:
        warn("%s." % errmsg)
        warning = errmsg
    elif ier in [0, 5, 6, 7, 8]:
        raise Exception("scipy.optimize.leastsq raises following error: '%s'." % errmsg)

    # 'nfev' number of function calls.
    f_count = infodict['nfev']

    # 'fvec': function evaluated at the output.
    fvec = infodict['fvec']
    #fvec_test = func(popt, times, values)

    # 'fjac': A permutation of the R matrix of a QR factorization of the final approximate Jacobian matrix, stored column wise. Together with ipvt, the covariance of the estimate can be approximated.
    # fjac = infodict['fjac']

    # 'qtf': The vector (transpose(q) * fvec).
    # qtf = infodict['qtf']

    # 'ipvt'  An integer array of length N which defines a permutation matrix, p, such that fjac*p = q*r, where r is upper triangular
    # with diagonal elements of nonincreasing magnitude. Column j of p is column ipvt(j) of the identity matrix.

    # 'pcov': The estimated covariance matrix of popt.
    # The diagonals provide the variance of the parameter estimate.

    # The reduced chi square: Take each "difference element, which could have been weighted" (O - E) and put to order 2. Sum them, and divide by number of degrees of freedom.
    # Calculated the (weighted) chi2 value.
    chi2 = numpy.sum( fvec**2 )

    # p is number of fitted parameters.
    p = len(x0)
    # n is number of degrees of freedom
    #n = N - p - 1
    n = N - p

    # The reduced chi square.
    chi2_red = chi2 / n

    # chi2_red >> 1 : indicates a poor model fit.
    # chi2_red >  1 : indicates that the fit has not fully captured the data (or that the error variance has been underestimated)
    # chi2_red = 1  : indicates that the extent of the match between observations and estimates is in accord with the error variance.
    # chi2_red <  1 : indicates that the model is 'over-fitting' the data: either the model is improperly fitting noise, or the error variance has been overestimated.

    absolute_sigma = True

    if pcov is None:
        # indeterminate covariance
        pcov = numpy.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(numpy.inf)
    elif not absolute_sigma:
        if N > p:
            pcov = pcov * chi2_red
        else:
            pcov.fill(inf)

    # To compute one standard deviation errors on the parameters, take the square root of the diagonal covariance.
    perr = numpy.sqrt(numpy.diag(pcov))

    # Return as standard from minfx.
    param_vector = popt
    param_vector_error = perr

    # Extract parameters from vector.
    amp, mu, sigma  = param_vector

    # Recalculate Full width at half maximum (FWHM)
    hm = 0.5 * amp
    fwhm = (2. * numpy.sqrt(2. * numpy.log(2.))) * sigma
    left_bin_x = mu - 0.5 * fwhm
    right_bin_x = mu + 0.5 * fwhm

    # Annotate the center.
    ax.annotate("%3.2f"%mu, xy=(mu, 0.0), xycoords='data', xytext=(mu, 0.25*amp), textcoords='data', size=8, horizontalalignment="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"), bbox=dict(boxstyle="round", facecolor="w"))

    # Annotate the Full width at half maximum.
    ax.annotate("", xy=(left_bin_x, hm), xycoords='data', xytext=(right_bin_x, hm), textcoords='data', arrowprops=dict(arrowstyle="<->", connectionstyle="arc3, rad=0"))
    ax.annotate("HM=%3.2f\nFWHM=%3.2f\nstd=%3.2f"%(hm, fwhm, sigma), xy=(mu, hm), xycoords="data", size=8, va="center", horizontalalignment="center", bbox=dict(boxstyle="round", facecolor="w"))

    # Calculate and plot the gauss values.
    gauss = func_gauss(params=param_vector, x=bincenters)
    ax.plot(bincenters, gauss, 'r-', label='gauss')

    # Calculate spread for sigma +/- 3. 99.7 % 
    sigma3_p = mu+3*sigma
    sigma3_m = mu-3*sigma

    ax.annotate("Sigma 3 plus", xy=(sigma3_p, 0.0), xycoords='data', xytext=(sigma3_p, 0.2*amp), textcoords='data', size=8, horizontalalignment="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"), bbox=dict(boxstyle="round", facecolor="w"))
    ax.annotate("Sigma 3 minus", xy=(sigma3_m, 0.0), xycoords='data', xytext=(sigma3_m, 0.2*amp), textcoords='data', size=8, horizontalalignment="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"), bbox=dict(boxstyle="round", facecolor="w"))

    sigma6_p = mu+6*sigma
    sigma6_m = mu-6*sigma

    ax.annotate("Sigma 6 plus", xy=(sigma6_p, 0.0), xycoords='data', xytext=(sigma6_p, 0.1*amp), textcoords='data', size=8, horizontalalignment="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"), bbox=dict(boxstyle="round", facecolor="w"))
    ax.annotate("Sigma 6 minus", xy=(sigma6_m, 0.0), xycoords='data', xytext=(sigma6_m, 0.1*amp), textcoords='data', size=8, horizontalalignment="center", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0"), bbox=dict(boxstyle="round", facecolor="w"))

    # Set limits.
    xlim = (mu-10*sigma, mu+25*sigma)
    ax.set_xlim(xlim)
    ylim = (0, bin_max_y)
    ax.set_ylim(ylim)

    # If show.
    if show:
        plt.show()

    # Return ax
    return ax, amp, mu, sigma, xlim, ylim


def linear_corr(x=None, y=None):
    """Calculate the linear correlation 'a', for the function y=a*x.  The function returns "a" and the sample correlation coefficient 'r_xy'.

    @keyword x:         The data for the X-axis.
    @type x:            float or numpy array.
    @keyword y:         The data for the Y-axis.
    @type y:            float or numpy array.
    @return:            The correlation 'a', and sample correlation coefficient 'r_xy'.
    @rtype:             float, float
    """

    # The correlation is.
    a = numpy.sum(x*y) / numpy.sum(x**2)

    # The sample correlation coefficient is.
    r_xy = numpy.sum(x*y) / numpy.sqrt(numpy.sum(x**2) * numpy.sum(y**2))

    return a, r_xy


def extract_rmsd(lines=None):
    """Extract showApod 'Noise Std Dev' for spectrum fourier transformed with NMRPipe.

    @keyword lines: The output lines from calling showApod
    @type leins:    list of sts
    @return:        The Noise Std Dev from line: 'REMARK Automated Noise Std Dev in Processed Data'
    @rtype:         float
    """

    # Loop over the lines
    found = False
    for line in lines:
        # Look for line with this remark.
        if line[:49] == 'REMARK Automated Noise Std Dev in Processed Data:':
            # The rest of the line is the rmsd.
            rmsd = float(line[49:].split()[0])
            return rmsd

    if not found:
        print(show_apod_lines)
        raise Exception("Could not find the line: 'REMARK Automated Noise Std Dev in Processed Data:', from the output of showApod.")


def make_nus_data(dir_from=None, dir_to=None, np=None, ni=None):
    """Create binary data, with FIDs from nls.hdr_3.

    @keyword dir_from:  Directory where original fid resides
    @type dir_from:     str
    @keyword dir_to:    Directory where to put the modified fid
    @type dir_from:     str
    @keyword np:        The number of points for the direct dimension in the original file.
    @type np:           int
    @keyword ni:        The number of points for the in-direct dimension in the original file.
    @type ni:           int
    """

    # Check that file exists:
    fname_or_fid = dir_from + os.sep + 'fid'
    test_fid = os.path.isfile(fname_or_fid)
    if not test_fid:
        print("\nThe original fid file cannot be found: %s\n"%fname_or_fid)
        raise Exception("The original fid file cannot be found")

    # Check that nls.hdr_3 exist in target dir
    fname_hdr3 = dir_to + os.sep + 'nls.hdr_3'
    test_hdr3 = os.path.isfile(fname_hdr3)
    if not test_hdr3:
        print("\nThe nls.hdr_3 file cannot be found: %s\n"%fname_hdr3)
        raise Exception("The nls.hdr_3 file cannot be found")

    # Open the NI from nls.hdr_3
    file_hdr3 = open(fname_hdr3, "r")
    ni_list = []
    for line in file_hdr3:
        ni_list.append(int(line))
    file_hdr3.close()

    # Calculate the bytes
    bytes = 4*np

    # Calculate the expected filesize.
    expected_file_size = 2*ni*(bytes+28)+32

    # Get the actual file size
    actual_file_size = os.path.getsize(fname_or_fid)
    test_file_size = expected_file_size == actual_file_size
    if not test_file_size:
        print("\nThe actual file size differs from the expected: %s vs: %s bytes.\n"%(actual_file_size, expected_file_size))
        raise Exception("The actual file size differs from the expected filesize.")

    # Open file for binary reading.
    file_fid_in = open(fname_or_fid, 'rb')

    # Open file for binary writing.
    fname_out_fid = dir_to + os.sep + 'fid'
    file_fid_out = open(fname_out_fid, 'wb')

    # First read the header and copy it.
    header = file_fid_in.read(32)
    file_fid_out.write(header)

    # Then loop over the list of NI.
    for ni_i in ni_list:
        # Calculate the offset position.
        seek_pos = 2*ni_i*(bytes+28)+32

        # Move the reader
        file_fid_in.seek(seek_pos)

        # Read header1, data1, header2, data2
        head1 = file_fid_in.read(28)
        data1 = file_fid_in.read(bytes)
        head2 = file_fid_in.read(28)
        data2 = file_fid_in.read(bytes)

        # Then write.
        file_fid_out.write(head1)
        file_fid_out.write(data1)
        file_fid_out.write(head2)
        file_fid_out.write(data2)

    # Close files.
    file_fid_in.close()
    file_fid_out.close()

    # Test the size of the written file.
    written_file_size = os.path.getsize(fname_out_fid)
    expected_written_file_size = 2*len(ni_list)*(bytes+28)+32
    test_file_size = expected_written_file_size == written_file_size
    if not test_file_size:
        print("\nThe actual written size differs from the expected: %s vs: %s bytes.\n"%(written_file_size, expected_written_file_size))
        raise Exception("The actual file size differs from the expected filesize.")


def read_spectrum(file=None):
    """Read the spectrum data.

    @keyword file:          The name of the file containing the spectrum.
    @type file:             str
    @return:                The nmrglue data dictionary, the universal dictionary, and the data as numpy array.
    @rtype:                 dic, dic, numpy array
    """

    # Open file
    dic, data = nmrglue.pipe.read(file)
    udic = nmrglue.pipe.guess_udic(dic, data)

    # Write out to txt.
    root, ext = os.path.splitext(file)
    if ext == ".ft2":
        numpy.savetxt('%s.out'%root, data.flatten(), delimiter=' ')

    # Return the nmrglue data object.
    return dic, udic, data


def write_fidSP_recFT(dir_from=None, dir_to=None, cur_ni=None):
    """Write fidSP.com and recFT.com, which tells how qMDD should process files.

    @keyword dir_from:  Directory where fid.com and nmrproc.com is located.
    @type dir_from:     str
    @keyword dir_to:    Directory where to put fidSP.com
    @type dir_to:       str
    @keyword cur_ni:    The number of in-direct increments to create the data matrix.
    @type cur_ni:       int
    """

    # Check that fid.com exist in original dir
    fname_fid_com = dir_from + os.sep + 'fid.com'
    test_fname_fid_com = os.path.isfile(fname_fid_com)
    if not test_fname_fid_com:
        print("\nThe fid.com file cannot be found: %s\n"%fname_fid_com)
        raise Exception("The fid.com file cannot be found")

    # Open the fid.com
    file_fid_com = open(fname_fid_com, "r")

    # Check that nmrproc.com exist in original dir
    fname_nmrproc_com = dir_from + os.sep + 'nmrproc.com'
    test_fname_nmrproc_com = os.path.isfile(fname_nmrproc_com)
    if not test_fname_nmrproc_com:
        print("\nThe nmrproc.com file cannot be found: %s\n"%fname_nmrproc_com)
        raise Exception("The nmrproc.com file cannot be found")

    # Open the nmrproc.com
    file_nmrproc_com = open(fname_nmrproc_com, "r")

    # Calculate dimension for matrix
    yT = cur_ni
    yN = 2*yT

    fidSP_lines = []
    for line in file_fid_com:
        # Skip lines with #
        if line[0] == "#":
            continue
        # Skip empty lines
        elif line == '\n':
            continue
        # If out is in line
        elif "-out" in line:
            continue
        # Skip lines with sleep
        elif "sleep" in line:
            continue

        # Now look for yN
        if  '-yN' in line:
            line_split = line.split()
            index_yN = line_split.index('-yN')
            line_split[index_yN+1] = str(yN)
            line = ' '.join(line_split) + '\n'
        # Now look for yT
        elif  '-yT' in line:
            line_split = line.split()
            index_yT = line_split.index('-yT')
            line_split[index_yT+1] = str(yT)
            line = ' '.join(line_split) + '\n'
        else:
            line_split = line.split()
            line = ' '.join(line_split) + '\n'

        # Append lines
        fidSP_lines.append(line)

    # Store nmrpipe commands depending.
    pipe_pre = []
    pipe_post = []
    store_pre = True

    for line in file_nmrproc_com:
        # Skip lines with #
        if line[0] == "#":
            continue
        # Skip empty lines
        elif line == '\n':
            continue
        # Skip reading in line
        elif 'nmrPipe -in' in line:
            continue
        # Skip reading out line
        elif '-out' in line:
            continue

        # Make swith to store
        if 'TP' in line:
            store_pre = False

        # Store lines
        if store_pre:
            pipe_pre.append(line)
        else:
            pipe_post.append(line)

    # Now write fidSP.com
    wfile_name_fidSP_com = dir_to + os.sep + 'fidSP.com'
    wfile_fidSP_com = open(wfile_name_fidSP_com, "w")

    for line in fidSP_lines + pipe_pre:
        wfile_fidSP_com.write(line)

    # Write additional
    wfile_fidSP_com.write(r'| pipe2xyz -z -out ft/data%03d.DAT -ov -nofs -verb')

    # Now write to recFT.com
    wfile_name_recFT_com = dir_to + os.sep + 'recFT.com'
    wfile_recFT_com = open(wfile_name_recFT_com, "w")

    recFT_lines_pre = [
    "#!/bin/tcsh -f",
    "echo '| ' in $0 $1",
    "if( $#argv < 1 ) then",
    'echo "Use: $0 <input pipe> <template for output spectrum>"',
    'echo "nmrPipe processing of YZ dimensions after MDD reconstruction"',
    "exit 1",
    "endif",
    "",
    "set ft4trec=$1",
    "if( $#argv > 1 ) set proc_out=$2",
    "",
    "if( ! -f $ft4trec ) then",
    "   ls $ft4trec",
    "   echo $0 failed",
    "   exit 2",
    "endif",
    "",
    "echo '|   Processing time domain MDD reconstruction '",
    "echo",
    "echo Processing Y dimensions",
    "showhdr $ft4trec",
    "cat $ft4trec                                        \\",
    ]

    recFT_lines_post = [
    "-ov -out $proc_out",
    "echo $proc_out ready",
    "exit",
    ]

    # Write
    for line in recFT_lines_pre:
        wfile_recFT_com.write(line + "\n")

    for line in pipe_post:
        wfile_recFT_com.write(line)

    for line in recFT_lines_post:
        wfile_recFT_com.write(line + "\n")

    # Close files.
    file_fid_com.close()
    file_nmrproc_com.close()
    wfile_fidSP_com.close()
    wfile_recFT_com.close()

    # Then make files executable.
    os.chmod(wfile_name_recFT_com, S_IRWXU|S_IRGRP|S_IROTH)


def write_proc(dir=None, FID=None, NUS_POINTS=None, FST_PNT_PPM=None, ROISW=None, SRSIZE=None, CEXP=None, MDDTHREADS=None, NCOMP=None, NITER=None, MDD_NOISE=None):
    """Create the nls.in file for nussampler.

    @keyword dir:           The directory where to create the nls.in file.
    @type dir:              str
    @keyword FID:           MDD param. The relative path to the directory with the fid file. The relative path should be wihtout ending of ".fid".
    @type FIDI:             str
    @keyword NUS_POINTS:    MDD param. How many points are sampled in the in-direct dimension.
    @type NUS_POINTS:       int
    @keyword FST_PNT_PPM:   MDD param. Define from where the region of interest starts in the direct dimension [ppm].
    @type FST_PNT_PPM:      float
    @keyword ROISW:         MDD param: Define sweep-width in ppm, to subtract from FST_PNT_PPM.
    @type ROISW:            float
    @keyword SRSIZE:        MDD param: Size of sub-region (ppm).
    @type SRSIZE:           float
    @keyword SRSIZE:        MDD param: Size of sub-region (ppm).
    @type SRSIZE:           str
    @keyword CEXP:          MDD param. Toggle R-MDD / MDD mode. For a dimension, with "y" time domain shape in the dimension is expected to be autoregressive. In other words, we assume that the FID in the dimension is a complex exponent. CEXP=y may be used, for example, for HNCO and HNcoCA experiments, but not for the NOESYs.
    @type CEXP:             str
    @keyword MDDTHREADS:    MDD param: Maximal number of parallel processes.
    @type MDDTHREADS:       int
    @keyword NCOMP:         MDD param: Number of components per sub-region
    @type NCOMP:            int
    @keyword NITER:         MDD param: Number of iteration in mddnmr.
    @type NITER:            int
    @keyword MDD_NOISE:     MDD param: Noise in mddnmr.
    @type MDD_NOISE:        float
    """

    # Open files.
    wfile_name_proc_comdd_before = dir + os.sep + 'proc_comdd_before.sh'
    wfile_proc_comdd_before = open(wfile_name_proc_comdd_before, "w")

    wfile_name_proc_comdd_after = dir + os.sep + 'proc_comdd_after.sh'
    wfile_proc_comdd_after = open(wfile_name_proc_comdd_after, "w")

    wfile_name_proc_mdd = dir + os.sep + 'proc_mdd.sh'
    wfile_proc_mdd = open(wfile_name_proc_mdd, "w")

    lines = [
    "#!/bin/tcsh",
    "setenv FID %s"%FID,
    "setenv fidSP fidSP.com",
    "setenv REC2FT recFT.com",
    "setenv in_file nls.in",
    "setenv selection_file nls.hdr_3",
    "setenv FST_PNT_PPM %s"%FST_PNT_PPM,
    "setenv ROISW %s"%ROISW,
    "setenv NUS_POINTS           %i"%NUS_POINTS,
    "setenv NUS_TABLE_OFFSET     0",
    "setenv SRSIZE               %s"%SRSIZE,
    "setenv CEXP                 %s"%CEXP,
    "setenv MDDTHREADS           %i"%MDDTHREADS,
    "setenv METHOD               MDD",
    "#MDD related parameters",
    "setenv NCOMP                %s"%NCOMP,
    "setenv NITER                %s"%NITER,
    "setenv MDD_NOISE            %s"%MDD_NOISE,
    ]

    # Write to files.
    for line in lines:
        wfile_proc_comdd_before.write(line + "\n")
        wfile_proc_comdd_after.write(line + "\n")
        wfile_proc_mdd.write(line + "\n")

    # Then write independent
    wfile_proc_comdd_before.write("setenv proc_out test.ft2" + "\n")
    wfile_proc_comdd_after.write("setenv proc_out test.ft2" + "\n")
    wfile_proc_mdd.write("setenv proc_out test.ft2" + "\n")
    wfile_proc_comdd_before.write("mddnmr4pipeN.sh 1 23" + "\n")
    wfile_proc_comdd_after.write("mddnmr4pipeN.sh 4 5" + "\n")
    wfile_proc_mdd.write("mddnmr4pipeN.sh 1 2 3 4 5" + "\n")

    # Close files.
    wfile_proc_comdd_before.close()
    wfile_proc_comdd_after.close()
    wfile_proc_mdd.close()

    # Then make files executable.
    os.chmod(wfile_name_proc_comdd_before, S_IRWXU|S_IRGRP|S_IROTH)
    os.chmod(wfile_name_proc_comdd_after, S_IRWXU|S_IRGRP|S_IROTH)
    os.chmod(wfile_name_proc_mdd, S_IRWXU|S_IRGRP|S_IROTH)

    # Now write a proc file, which is only for FT
    wfile_name_proc_FT = dir + os.sep + 'proc_FT.sh'
    wfile_proc_FT = open(wfile_name_proc_FT, "w")

    lines = [
    "#!/bin/tcsh",
    "setenv FID %s"%FID,
    "setenv fidSP fidSP.com",
    "setenv REC2FT recFT.com",
    "setenv in_file nls.in",
    "setenv selection_file nls.hdr_3",
    "setenv FST_PNT_PPM %s"%FST_PNT_PPM,
    "setenv ROISW %s"%ROISW,
    "setenv proc_out test_FT.ft2",
    "setenv NUS_POINTS           %i"%NUS_POINTS,
    "setenv NUS_TABLE_OFFSET     0",
    "setenv MDDTHREADS           %i"%MDDTHREADS,
    "setenv METHOD               FT",
    "mddnmr4pipeN.sh 1 2 3 4 5",
    ]

    # Write to file.
    for line in lines:
        wfile_proc_FT.write(line + "\n")

    # Close files.
    wfile_proc_FT.close()

    # Then make files executable.
    os.chmod(wfile_name_proc_FT, S_IRWXU|S_IRGRP|S_IROTH)


def write_nls_in(dir=None, NI=None, NIMAX=None, CEXP=None, T2=None, SW=None):
    """Create the nls.in file for nussampler.

    @keyword dir:           The directory where to create the nls.in file.
    @type dir:              str
    @keyword NI:            The number of increments to use out the full range of increments.
    @type NI:               int
    @keyword NIMAX:         The maximum number of increments in the in-indirect dimension.
    @type NIMAX:            int
    @keyword CEXP:          The maximum number of increments in the in-indirect dimension.
    @type CEXP:             int
    @keyword T2:            The spin-spin relaxation time, the expected time constant characterizing the signal decay in in-direct dimension [s].
    @type T2:               float
    @keyword SW:            The spectral width in in-directly detected dimension [Hz].
    @type SW:               float
    """

    # Open file
    wfile_name = dir + os.sep + 'nls.in'

    # If the file exists
    if os.path.exists(wfile_name):
        print(" nls.in file already exist. Skipping creating new, and nls.hdr_3. %s"%wfile_name)
        #raise Exception("nls.in file exist.")
        return        

    wfile = open(wfile_name, "w")

    lines = [
    "NDIM  2",
    "SPARSE y",
    "sptype shuffle",
    "seed   %i"%random.randint(1,100000),
    "CEXP   %s"%CEXP,
    "NIMAX  %i  1"%NIMAX,
    "NI     %i  1"%NI,
    "SW     %s 1"%SW,
    "T2      %s 1"%T2,
    ]

    # Write file
    for line in lines:
        wfile.write(line + "\n")

    # Close file.
    wfile.close()

    # Call the nussampler on the nls.in file.
    args = ('nussampler', wfile_name)

    # Call nussampler and get return code.
    returncode, line_split = call_prog(args=args)

def write_data(out=None, headings=None, data=None, sep=None):
    """Write out a table of the data to the given file handle.

    @keyword out:       The file handle to write to.
    @type out:          file handle
    @keyword headings:  The optional headings to print out.
    @type headings:     list of str or None
    @keyword data:      The data to print out.
    @type data:         list of list of str
    @keyword sep:       The column separator which, if None, defaults to whitespace.
    @type sep:          str or None
    """

    # No data to print out.
    if data in [None, []]:
        return

    # The number of rows and columns.
    num_rows = len(data)
    num_cols = len(data[0])

    # Pretty whitespace formatting.
    if sep == None:
        # Determine the widths for the headings.
        widths = []
        for j in range(num_cols):
            if headings != None:
                if j == 0:
                    widths.append(len(headings[j]) + 2)
                else:
                    widths.append(len(headings[j]))

            # No headings.
            else:
                widths.append(0)

        # Determine the maximum column widths for nice whitespace formatting.
        for i in range(num_rows):
            for j in range(num_cols):
                size = len(data[i][j])
                if size > widths[j]:
                    widths[j] = size

        # Convert to format strings.
        formats = []
        for j in range(num_cols):
            formats.append("%%-%ss" % (widths[j] + 4))

        # The headings.
        if headings != None:
            out.write(formats[0] % ("# " + headings[0]))
            for j in range(1, num_cols):
                out.write(formats[j] % headings[j])
            out.write('\n')

        # The data.
        for i in range(num_rows):
            # The row.
            for j in range(num_cols):
                out.write(formats[j] % data[i][j])
            out.write('\n')

    # Non-whitespace formatting.
    else:
        # The headings.
        if headings != None:
            out.write('#')
            for j in range(num_cols):
                # The column separator.
                if j > 0:
                    out.write(sep)

                # The heading.
                out.write(headings[j])
            out.write('\n')

        # The data.
        for i in range(num_rows):
            # The row.
            for j in range(num_cols):
                # The column separator.
                if j > 0:
                    out.write(sep)

                # The heading.
                out.write(data[i][j])
            out.write('\n')


def contour_plot(dic=None, udic=None, data=None, contour_start=30000., contour_num=20, contour_factor=1.20, ppm=True, show=False, table=None):
    """Plot the spectrum as contour plot.

    @keyword dic:               The data dictionary, from nmrglue.
    @type dic:                  dict
    @keyword udic:              The universal dictionary, from nmrglue.
    @type udic:                 dict
    @keyword data:              The spectrum data as 2D numpy array.
    @type data:                 2D numpy array
    @keyword contour_start:     Contour level start value
    @type contour_start:        float
    @keyword contour_num:       Number of contour levels
    @type contour_num:          int
    @keyword contour_factor:    Scaling factor between contour levels
    @type contour_factor:       float
    @keyword ppm:               A flag which if True will make the plot in ppm scale. Else it is in points.
    @type ppm:                  bool
    @keyword show:              A flag which if True will make a call to matplotlib.pyplot.show().
    @type show:                 bool
    @keyword table:             Peak table from nmrglue.analysis.peakpick.pick.
    @type table:                recarray
    @return:                    The matplotlib.axes.AxesSubplot class, which can be manipulated to add additional text to the axis.
    @rtype:                     matplotlib.axes.AxesSubplot
    """

    # Setup plot parameters
    # http://www.physics.ox.ac.uk/Users/msshin/science/code/matplotlib_cm/
    # contour map (colors to use for contours)
    #cmap = matplotlib.cm.Blues_r
    #cmap = matplotlib.cm.bwr
    #cmap = matplotlib.cm.RdBu_r
    #cmap = matplotlib.cm.coolwarm
    #cmap = matplotlib.cm.brg
    cmap = matplotlib.cm.seismic_r

    # Calculate contour levels
    cl_pos = contour_start * contour_factor ** numpy.arange(contour_num)
    cl_neg = - contour_start * contour_factor ** numpy.arange(contour_num)[::-1]

    cl = numpy.concatenate((cl_neg, cl_pos))

    # Create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the contours

    # Plot in ppm scale
    if ppm:
        # make ppm scales
        uc_dim1 = nmrglue.pipe.make_uc(dic, data, dim=1)
        ppm_dim1 = uc_dim1.ppm_scale()
        ppm_dim1_0, ppm_dim1_1 = uc_dim1.ppm_limits()
        uc_dim0 = nmrglue.pipe.make_uc(dic, data, dim=0)
        ppm_dim0 = uc_dim0.ppm_scale()
        ppm_dim0_0, ppm_dim0_1 = uc_dim0.ppm_limits()

        ax.contour(data.T, cl, cmap=cmap, extent=(ppm_dim0_0, ppm_dim0_1, ppm_dim1_0, ppm_dim1_1))

        # Decorate
        ax.set_title("Spectrum at contour %3.1f"%contour_start)

        ax.set_ylabel("%s (ppm)"%udic[0]['label'])
        ax.set_xlabel("%s (ppm)"%udic[1]['label'])
        lim_dim1 = [ppm_dim1_0, ppm_dim1_1]
        lim_dim0 = [ppm_dim0_0, ppm_dim0_1]
        ax.set_ylim(max(lim_dim1), min(lim_dim1))
        ax.set_xlim(max(lim_dim0), min(lim_dim0))

        # If point table scale.
        if table != None:
            x_ppm = uc_dim0.unit(table['Y_AXIS'],"ppm")
            y_ppm = uc_dim1.unit(table['X_AXIS'],"ppm")

            ax.plot(x_ppm, y_ppm, 'gx', alpha=0.5)

    # If show.
    if show:
        plt.show()

    # Return ax
    return ax


def write_peak_list(filename=None, x_axis_pts=None, y_axis_pts=None, x_axis_ppm=None, y_axis_ppm=None, int_ref=None, list_int_cov=None, pct_change=2.):

    # Define headings
    filew = open(filename, "w")

    filew.write("VARS INDEX X_AXIS Y_AXIS X_PPM Y_PPM X1 X3 Y1 Y3 HEIGHT"+'\n')
    filew.write(r"FORMAT %5d %9.3f %9.3f %8.3f %8.3f %4d %4d %4d %4d %+e"+'\n')

    # Define proportion change
    prop_ch_p = 1.0 + pct_change / 100.
    prop_ch_m = 1.0 - pct_change / 100.

    # Find number of peaks
    n_peaks = len(int_ref)

    for i in range(len(x_axis_pts)):
        index = "%i" % i
        x_pts = "%i"% int(x_axis_pts[i])
        y_pts = "%i"% int(y_axis_pts[i])
        x_ppm = "%3.3f"% x_axis_ppm[i]
        y_ppm = "%3.3f"% y_axis_ppm[i]
        height = "%1.5e"% int_ref[i]

        string = "%3s %4s %4s %6s %6s %4s %4s %4s %4s %8s" % (index, x_pts, y_pts, x_ppm, y_ppm, x_pts, x_pts, y_pts, y_pts, height)

        if list_int_cov != None:
            for int_cov in list_int_cov:
                height_cov = int_cov[i]
                int_prop = height_cov / int_ref[i]
                int_prop_s = "%1.5f" % int_prop


                # Test for pct change per peak
                if int_prop > prop_ch_p or int_prop < prop_ch_m:
                    mark = "x"
                else:
                    mark = " "

                # Count all changes
                int_prop_array = int_cov/int_ref
                mask_pct_change = masked_where( numpy.logical_or( int_prop_array < prop_ch_m, prop_ch_p < int_prop_array) , int_prop_array)
                peaks_change = int_prop_array[mask_pct_change.mask]

                if type(peaks_change) != numpy.ndarray:
                    n_peaks_change = 0
                else:
                    n_peaks_change = len(peaks_change)

                pct_peaks_change = float(n_peaks_change) / float(n_peaks) * 100.

                string += "| %8s %s %2.1f%%" % (int_prop_s, mark, pct_peaks_change)

        filew.write(string + "\n")

    # Close file.
    filew.close()


if __name__ == "__main__":
    # Check files are available.
    files = ['fid', 'fid.com', 'nmrproc.com', 'test.fid']
    check_files(files=files)

    # Create dictionary to store results
    res_dic = {}
    res_dic['input'] = {} 

    # Loop over args
    print("Input arguments")
    for arg in vars(input_args):
        print(arg, getattr(input_args, arg), type(getattr(input_args, arg)))
        res_dic['input']['arg'] = getattr(input_args, arg)

    # With nmrglue, read the binary transformed spectrum to get information.
    dic, udic, data = read_spectrum(file='test.fid')

    # Get the directory to create
    cwd = os.getcwd()
    now = time.localtime()
    #startdir = os.getcwd() + os.sep + "qNLS_%i%02i%02i_%02i%02i%02i" % (now[0],now[1], now[2], now[3], now[4], now[5])
    startdir = cwd + os.sep + "qNLS_%i%02i%02i" % (now[0],now[1], now[2])

    # Get the number of points in the direct dimension
    np = udic[1]['size']*2

    # Get the number of real + imaginary reconstructed points. 
    td1 = udic[0]['size']

    # Get the number of increments pairs in the in-direct dimension.
    ni = td1 / 2

    # Calculate R2
    R2 = 1.0 / input_args.T2

    # Get the obsfreq1 in MHz.
    obsfreq1 = udic[0]['obs']

    # Get the sw in the in-direct dimension, in Hz
    sw = udic[0]['sw']

    # Calculate the sw in ppm.
    sw_ppm = sw / obsfreq1

    # Get path to dpoisson7.1.1. NUSScore script from Peter E. Wright.
    path_dpoisson = os.path.dirname(os.path.realpath(__file__))+os.sep+"Aoto_Fenwick_Kroon_Wright_2014_NUSScore_linux_32bit-release"+os.sep+"dpoisson7.1.1"

    test = ni < 400
    if not test:
        print("\nNumber of increments in the in-direct dimension is to high: %i\n"%ni)
        raise Exception("NI is found to be to high. As a safety measure, it is expected to be under 400.")

    # Create list with nearest intergers of NI, from sampling fractions.
    sf_arr = numpy.array(input_args.sf)
    # Make NI array
    ni_arr = ni * sf_arr / 100.
    # Convert to nearest even ni.
    ni_arr = numpy.around(ni_arr/2)*2
    # Convert to int16
    ni_arr = numpy.array( ni_arr, dtype=numpy.int16 )

    # Then create the directories.
    # Loop over the sampling fractions.
    sf_dirs = []
    for sf in input_args.sf:
        cur_dir = '%02dpct'%sf
        sf_dirs.append(cur_dir)

    # Make list of ni and proc directories to create.
    ni_dirs = []
    proc_dirs = []

    for j in range(1, input_args.N + 1):
        cur_dir = '%02d.fid'%j
        ni_dirs.append(cur_dir)

        cur_dir = '%02d.proc'%j
        proc_dirs.append(cur_dir)

    # Store to res_dic
    res_dic['sf_arr'] = sf_arr
    res_dic['ni_arr'] = ni_arr
    res_dic['sf_dirs'] = sf_dirs
    res_dic['ni_dirs'] = ni_dirs
    res_dic['proc_dirs'] = proc_dirs

    # Loop over sample fraction level.
    for i, sf_dir in enumerate(sf_dirs):
        cur_ni = ni_arr[i]

        # Store to dic
        res_dic[sf_dir] = {}
        res_dic[sf_dir]['cur_ni'] = cur_ni

        # Create dir for dpoisson
        out_dpoisson = startdir + os.sep + 'NUSScore' + os.sep + sf_dir
        if not os.path.exists(out_dpoisson):
            os.makedirs(out_dpoisson)

        # Calculate the current coverage
        cur_cov = float(cur_ni)/float(ni)

        # Create range of  schdedules with dpoisson
        print("Creating %i NUSScedules with dpoisson7.1.1"%input_args.N_NUS_SCHEDULES)
        poisson_args = ["%1.7f"%cur_cov, "%i"%td1, "1", "%2.1f"%R2, "1", "0", "0", "%2.3f"%obsfreq1, "1", "%2.2f"%sw_ppm, "1", "0", "%i"%input_args.N_NUS_SCHEDULES, "1", "%s"%out_dpoisson]
        print("Created with args:", poisson_args)
        returncode, line_split = call_prog(args=[path_dpoisson] + poisson_args, verbose=True)

        # Now get the filename for the best schedule.
        fname_nus_top_out = out_dpoisson + os.sep + "score" + os.sep + "nus_top.out"

        # Open the file.
        file_nus_top_out = open(fname_nus_top_out, "r")
        lines_nus_top_out = file_nus_top_out.readlines()

        # Close the file.
        file_nus_top_out.close()

        # Get filename for best nus schedule
        fname_best_nus = out_dpoisson + os.sep + "lists" + os.sep + lines_nus_top_out[3].split()[0]
        #file_best_nus = open(fname_best_nus, "r")
        #lines_best_nus = file_best_nus.readlines()
        #print(lines_best_nus)

        # Create dir for restoring.
        create_sf_dir = startdir + os.sep + sf_dir
        if not os.path.exists(create_sf_dir):
            os.makedirs(create_sf_dir)

        # Then create reference dir.
        create_ni_ref_dir = create_sf_dir + os.sep + '00_ref.fid'
        if not os.path.exists(create_ni_ref_dir):
            os.makedirs(create_ni_ref_dir)
        # Make a full ni.
        write_nls_in(dir=create_ni_ref_dir, NI=ni, NIMAX=ni, T2=input_args.T2, SW=sw)

        # Then create the ni shuffled binary data
        make_nus_data(dir_from=cwd, dir_to=create_ni_ref_dir, np=np, ni=ni)

        # Now write: proc_FT.sh, proc_comdd_before.sh, proc_comdd_after.sh, fidSP.com, recFT.com
        create_proc_ref_dir = create_sf_dir + os.sep + '00_ref.proc'
        if not os.path.exists(create_proc_ref_dir):
            os.makedirs(create_proc_ref_dir)

        # Write the proc files. 'proc_comdd_before.sh', 'proc_comdd_after.sh'
        # The first one creates the MDD data
        # The second one is for coMDD solving.
        write_proc(dir=create_proc_ref_dir, FID='../00_ref', NUS_POINTS=ni, FST_PNT_PPM=input_args.FST_PNT_PPM, ROISW=input_args.ROISW, SRSIZE=input_args.SRSIZE, CEXP=input_args.CEXP, MDDTHREADS=input_args.MDDTHREADS, NCOMP=input_args.NCOMP, NITER=input_args.NITER, MDD_NOISE=input_args.MDD_NOISE)

        # Write fidSP.com and recFT.com file.
        write_fidSP_recFT(dir_from=cwd, dir_to=create_proc_ref_dir, cur_ni=ni)

        # Copy over nls.in and nls.hdr_3.
        shutil.copy(create_ni_ref_dir + os.sep + 'nls.in', create_proc_ref_dir)
        shutil.copy(create_ni_ref_dir + os.sep + 'nls.hdr_3', create_proc_ref_dir)

        # If MDD directory is missing, call initial script to create MDD files.
        if not os.path.isdir(create_proc_ref_dir + os.sep + 'MDD'):
            # Change current directory
            os.chdir(create_proc_ref_dir)
            # Call script to create files.
            call_prog(args=['proc_comdd_before.sh'])
            # Change back again
            os.chdir(cwd)

        # Producefile for reference.
        path_ref_REF_ft2file = create_proc_ref_dir + os.sep + 'test_FT.ft2'
        if not os.path.exists(path_ref_REF_ft2file):
            # Change current directory
            os.chdir(create_proc_ref_dir)
            # Call script to create files.
            call_prog(args=['proc_FT.sh'])
            # Change back again
            os.chdir(cwd)

        # Now read data for reference
        returncode, line_split = call_prog(args=['showApod', path_ref_REF_ft2file], verbose=False)
        rmsd = extract_rmsd(lines=line_split)
        res_dic[sf_dir]['ref'] = {}
        res_dic[sf_dir]['ref']['rmsd'] = rmsd

        # With nmrglue, read the fourier transformed spectrum to get information.
        dic_ref, udic_ref, data_ref = read_spectrum(file=path_ref_REF_ft2file)
        res_dic[sf_dir]['ref']['dic'] = dic_ref
        res_dic[sf_dir]['ref']['udic'] = udic_ref
        res_dic[sf_dir]['ref']['data'] = data_ref

        # Now do a peak list
        table = nmrglue.analysis.peakpick.pick(data=data_ref, pthres=20*rmsd, nthres=None, algorithm='connected', est_params=False, cluster=False, table=True)
        #table = nmrglue.analysis.peakpick.pick(data=data_ref, pthres=20*rmsd, nthres=None, algorithm='downward', est_params=False, cluster=False, table=True)

        # Now convert points to ppm.
        uc_dim0 = nmrglue.pipe.make_uc(dic_ref, data_ref, dim=0)
        uc_dim1 = nmrglue.pipe.make_uc(dic_ref, data_ref, dim=1)
        y_axisppm = uc_dim0.unit(table['Y_AXIS'], "ppm")
        x_axisppm = uc_dim1.unit(table['X_AXIS'], "ppm")

        # Try a contour plot.
        contour_plot(dic=dic_ref, udic=udic_ref, data=data_ref, contour_start=20*rmsd, contour_num=10, contour_factor=1.20, ppm=True, show=False, table=table)
        png_path = startdir + os.sep + "spec_FT_ref.png"
        plt.savefig(png_path, format='png', dpi=600)
        # Close figure.
        plt.close("all")

        # Make a histogram
        ax, amp, mu, sigma, xlim_ref, ylim_ref = hist_plot(ndarray=data_ref, show=False)
        res_dic[sf_dir]['ref']['hist'] = [amp, mu, sigma]

        png_path = startdir + os.sep + "hist_FT_ref.png"
        plt.savefig(png_path, format='png', dpi=600)
        # Close figure.
        plt.close("all")

        # Flatten data
        data_ref_flat = data_ref.flatten()

        # Make selection masks
        sigma3 = mu + 3*sigma
        sigma10 = mu + 10*sigma
        sigma100 = mu + 100*sigma
        sigma1000 = mu + 1000*sigma

        mask_to_sigma3 = masked_where(data_ref_flat < sigma3, data_ref_flat)
        mask_3_to_10 = masked_where( numpy.logical_and( sigma3 <= data_ref_flat, data_ref_flat < sigma10) , data_ref_flat)
        mask_10_to_100 = masked_where( numpy.logical_and( sigma10 <= data_ref_flat, data_ref_flat < sigma100) , data_ref_flat)
        mask_100_to_1000 = masked_where( numpy.logical_and( sigma100 <= data_ref_flat, data_ref_flat < sigma1000 ) , data_ref_flat)
        mask_from_1000 = masked_where( sigma1000 <= data_ref_flat, data_ref_flat)

        ## Collect masks for graphs and their hex color.
        sn_masks = [
        [r'$I < 3\sigma$', "#FC0000", mask_to_sigma3, 'to_s3'], 
        [r'$3\sigma \leq I < 10\sigma$', "#F0FC00", mask_3_to_10, 's3_to_10'],
        [r'$10\sigma \leq I < 100\sigma$', "#0DFC00", mask_10_to_100, 's10_to_s100'],
        [r'$100\sigma \leq I < 1000\sigma$', "#00FCF8", mask_100_to_1000, 's100_to_s1000'],
        [r'$1000\sigma \leq I\sigma$', "#FC00F8", mask_from_1000, 'from_s1000'],
        ]

        # Now make report for pct
        pct_results_name = startdir + os.sep + "pct_%s_results.txt"%(sf_dir)
        pct_results = open(pct_results_name, 'w')

        # Collect header
        headers = []

        # Collect data
        datacsv = []

        sn_masks_used = []
        for label, color, sel_mask, dickey in sn_masks:
            data_ref_mask = data_ref_flat[sel_mask.mask]

            if type(data_ref_mask) != numpy.ndarray:
                continue
            pct = float(len(data_ref_mask)) / float(len(data_ref_flat)) * 100.
            sn_masks_used.append([label, color, sel_mask, dickey, pct])

            headers.append(dickey)
            datacsv.append("%2.1f"%pct)

        # Write data
        write_data(out=pct_results, headings=headers, data=[datacsv])
        pct_results.close()

        # Then create ni dirs.
        for j, ni_dir in enumerate(ni_dirs):
            create_ni_dir = create_sf_dir + os.sep + ni_dir
            if not os.path.exists(create_ni_dir):
                os.makedirs(create_ni_dir)
            # Make for ni.
            write_nls_in(dir=create_ni_dir, NI=cur_ni, NIMAX=ni, CEXP=input_args.CEXP, T2=input_args.T2, SW=sw)

            # Then create the ni shuffled binary data
            make_nus_data(dir_from=cwd, dir_to=create_ni_dir, np=np, ni=ni)

            # Now write: proc_FT.sh, proc_comdd_before.sh, proc_comdd_after.sh, fidSP.com, recFT.com
            proc_dir = proc_dirs[j]
            create_proc_dir = create_sf_dir + os.sep + proc_dir
            if not os.path.exists(create_proc_dir):
                os.makedirs(create_proc_dir)

            # Write the proc files. 'proc_comdd_before.sh', 'proc_comdd_after.sh'
            # The first one creates the MDD data
            # The second one is for coMDD solving.
            FID = '..' + os.sep + proc_dir.split('.proc')[0]
            write_proc(dir=create_proc_dir, FID=FID, NUS_POINTS=cur_ni, FST_PNT_PPM=input_args.FST_PNT_PPM, ROISW=input_args.ROISW, SRSIZE=input_args.SRSIZE, CEXP=input_args.CEXP, MDDTHREADS=input_args.MDDTHREADS, NCOMP=input_args.NCOMP, NITER=input_args.NITER, MDD_NOISE=input_args.MDD_NOISE)

            # Write fidSP.com and recFT.com file.
            write_fidSP_recFT(dir_from=cwd, dir_to=create_proc_dir, cur_ni=cur_ni)

            # Copy over nls.in and nls.hdr_3.
            shutil.copy(create_ni_dir + os.sep + 'nls.in', create_proc_dir)
            shutil.copy(create_ni_dir + os.sep + 'nls.hdr_3', create_proc_dir)

            # If MDD directory is missing, call initial script to create MDD files.
            if not os.path.isdir(create_proc_dir + os.sep + 'MDD'):
                # Change current directory
                os.chdir(create_proc_dir)
                # Call script to create files.
                call_prog(args=['proc_comdd_before.sh'])
                # Change back again
                os.chdir(cwd)

        # Then create coMDD dir
        coMDD_dir = create_sf_dir + os.sep + 'coMDD'
        if not os.path.isdir(coMDD_dir):
            os.makedirs(coMDD_dir)

        # Then create coMDD.hd.
        wfile_name_coMDD_hd = coMDD_dir + os.sep + 'coMDD.hd'
        wfile_coMDD_hd = open(wfile_name_coMDD_hd, "w")

        # Collect all proc dirs
        all_proc_dirs = ['00_ref.proc'] + proc_dirs
        res_dic[sf_dir]['all_proc_dirs'] = all_proc_dirs

        for j, proc_dir in enumerate(all_proc_dirs):
            FID = proc_dir.split('.proc')[0]
            proc_pos = '..' + os.sep + proc_dir + os.sep + 'MDD' + os.sep + r'region%02d.mddH'
            # It is the reading of CEXP in nls.in, that makes coMDD/hd01.mdd be rmdd.
            nls_in_pos = '..' + os.sep + proc_dir + os.sep + 'nls.in'
            string = "%i %s 1.0 %s %s 1 2"%(j, FID, proc_pos, nls_in_pos)

            wfile_coMDD_hd.write(string + "\n")

        wfile_coMDD_hd.close()

        # Then get number of regions, and convert file to HD data assembling.
        nreg = 0
        region_file = open(create_proc_ref_dir + os.sep + 'regions.runs', "r")
        hd_region_file = open(coMDD_dir + os.sep + 'regions.runs', "w")
        for line in region_file:
            if "mddsolver" in line:
                nreg += 1

            wstring = line.replace("./MDD/region", "hd");            
            hd_region_file.write(wstring)

        region_file.close()
        hd_region_file.close()

        # Then do HD data assembling
        list_files = glob.glob(coMDD_dir+ os.sep + 'hd*.mdd')
        if len(list_files) != nreg:
            print("HD data assembling")
            # Change current directory
            os.chdir(coMDD_dir)
            # Call script to create files.
            call_prog(args=['setHD', 'coMDD.hd', 'mdd', '%i'%nreg, r'hd%02d.mdd'])
            # Change back again
            os.chdir(cwd)

        # Test if do MDD calculation
        list_files = glob.glob(coMDD_dir+ os.sep + 'hd*.res')
        if len(list_files) != nreg:
            # Then do MDD calculation
            print("MDD calculation")
            # Change current directory
            os.chdir(coMDD_dir)
            # Call script to create files.
            call_prog(args=['queMM.sh', 'regions.runs'])
            # Change back again
            os.chdir(cwd)

        # Make HD data disassembling
        # Change current directory
        os.chdir(coMDD_dir)
        # Call script to create files.
        for j, proc_dir in enumerate(all_proc_dirs):
            proc_MDD_dir = create_sf_dir + os.sep + proc_dir + os.sep + 'MDD'
            list_files = glob.glob(proc_MDD_dir + os.sep + 'region*.res')
            if len(list_files) != nreg:
                print("HD data disassembling in %s"%proc_MDD_dir)
                proc_pos = '..' + os.sep + proc_dir + os.sep + 'MDD' + os.sep + r'region%02d.res'
                args = ['setHD', 'coMDD.hd', 'res', '%i'%nreg, "%s"%proc_pos, r"hd%02d.res", str(j)]
                call_prog(args=args)
        # Change back again
        os.chdir(cwd)

        # Then assemble data
        for j, proc_dir in enumerate(all_proc_dirs):
            cur_proc_dir = create_sf_dir + os.sep + proc_dir
            path_ft2_file = cur_proc_dir + os.sep + 'test.ft2'
            if not os.path.exists(path_ft2_file):
                # Change current directory
                os.chdir(cur_proc_dir)
                # Call script to create files.
                print("Assembling data in: %s"%cur_proc_dir)
                call_prog(args=['proc_comdd_after.sh'])
                # Change back again
                os.chdir(cwd)
            else:
                print("File exists. I do not produce .ft2 file again.: %s"%path_ft2_file)

        print("Now making figures and reports.")

        # First find the linear difference between ref, and the full data.
        path_ref_FULL_ft2file = create_proc_ref_dir + os.sep + 'test.ft2'

        # With nmrglue, read the fourier transformed spectrum to get information.
        dic_ref_full, udic_ref_full, data_ref_full = read_spectrum(file=path_ref_FULL_ft2file)
        a_dev, r_xy_dev = linear_corr(x=data_ref_flat, y=data_ref_full.flatten())

        # Get the RMSD from showApod
        returncode, line_split = call_prog(args=['showApod', path_ref_FULL_ft2file], verbose=False)
        rmsd_FULL_ft2file = extract_rmsd(lines=line_split)

        # Measure the intensity
        data_ref_full_int = data_ref_full[table['Y_AXIS'].astype(int), table['X_AXIS'].astype(int)]

        # Collect proc ints
        proc_ints = []

        # Then collect showApod rmsd
        # Get showApod
        for j, proc_dir in enumerate(all_proc_dirs):
            cur_proc_dir = create_sf_dir + os.sep + proc_dir
            path_ft2_file = cur_proc_dir + os.sep + 'test.ft2'

            returncode, line_split = call_prog(args=['showApod', path_ft2_file], verbose=False)
            rmsd = extract_rmsd(lines=line_split)

            # Store to results dic
            res_dic[sf_dir][proc_dir] = {}
            res_dic[sf_dir][proc_dir]['rmsd'] = rmsd
        
            # With nmrglue, read the fourier transformed spectrum to get information.
            dic_cur, udic_cur, data_cur = read_spectrum(file=path_ft2_file)
            res_dic[sf_dir][proc_dir]['dic'] = dic_cur
            res_dic[sf_dir][proc_dir]['udic'] = udic_cur
            res_dic[sf_dir][proc_dir]['data'] = data_cur

            # Measure the intensity of peaks.
            data_cur_int = data_cur[table['Y_AXIS'].astype(int), table['X_AXIS'].astype(int)]
            proc_ints.append(data_cur_int)

            # Try a contour plot.
            contour_plot(dic=dic_cur, udic=udic_cur, data=data_cur, contour_start=20*rmsd_FULL_ft2file, contour_num=10, contour_factor=1.20, ppm=True, show=False, table=table)
            png_path = startdir + os.sep + "spect_%s_%s.png"%(sf_dir, proc_dir)
            plt.savefig(png_path, format='png', dpi=600)
            # Close figure.
            plt.close("all")

            # Make a residual intensity spectrum.
            data_resi = data_cur - data_ref_full
            path_resi_spec = cur_proc_dir + os.sep + 'test_resi.ft2'
            nmrglue.fileio.pipe.write(filename=path_resi_spec, dic=dic_cur, data=data_resi, overwrite=True)

            # Try a contour plot.
            contour_plot(dic=dic_cur, udic=udic_cur, data=data_resi, contour_start=6*rmsd_FULL_ft2file, contour_num=10, contour_factor=1.20, ppm=True, show=False, table=table)
            png_path = startdir + os.sep + "resi_spect_%s_%s.png"%(sf_dir, proc_dir)
            plt.savefig(png_path, format='png', dpi=600)
            # Close figure.
            plt.close("all")

            # Make a histogram
            ax, amp, mu, sigma, xlim, ylim = hist_plot(ndarray=data_cur, show=False)
            res_dic[sf_dir][proc_dir]['hist'] = [amp, mu, sigma]

            # Set same limits as ref
            ax.set_xlim(xlim_ref)
            ax.set_ylim(ylim_ref)

            png_path = startdir + os.sep + "hist_%s_%s.png"%(sf_dir, proc_dir)
            if not os.path.isfile(png_path):
                plt.savefig(png_path, format='png', dpi=600)
            # Close figure.
            plt.close("all")
            print("Made figure: %s"%png_path)

            # Make a correlation plot
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Flatten data
            data_cur_flat = data_cur.flatten()

            # Try get the linear correlation
            a, r_xy = linear_corr(x=data_ref_flat, y=data_cur_flat)
            res_dic[sf_dir][proc_dir]['corr'] = [a, r_xy]

            # Make line.
            line = numpy.array( [data_ref_flat.min(), data_ref_flat.max()] )
            ax.plot(line, line, 'g-', linewidth=0.5, label='ref vs ref.')
            ax.plot(line, line*a, 'b-', linewidth=0.2, label='Linear')

            # Collect for different signal levels.
            res_dic[sf_dir][proc_dir]['corr_s'] = {}

            # Loop over data mask
             #ax.plot(data_ref_flat, data_cur_flat, 'b.', markersize=2, label='all int')
            for label, color, sel_mask, dickey, pct in sn_masks_used:
                data_ref_mask = data_ref_flat[sel_mask.mask]
                data_cur_mask = data_cur_flat[sel_mask.mask]
                a_mask, r_xy_mask = linear_corr(x=data_ref_mask, y=data_cur_mask)
                res_dic[sf_dir][proc_dir]['corr_s'][dickey] = [a_mask, r_xy_mask]

                deviation = data_cur_mask - a_dev * data_ref_mask
                rmsd = numpy.sqrt(numpy.mean(numpy.square(deviation)))
               
                ax.plot(data_ref_mask, data_cur_mask, '.', color=color, markersize=2, label='%s , pct=%2.1f, a=%1.2f, r_xy^2=%3.4f, rmsd=%3.4f'%(label, pct, a_mask, r_xy_mask**2, rmsd))

            # Set text.
            ax.set_xlabel("All spectrum intensities for reference")
            ax.set_ylabel("All spectrum intensities for method")
            ax.annotate("a=%3.6f\nr_xy=%3.6f\nr_xy^2=%3.6f"%(a, r_xy, r_xy**2), xy=(data_ref_flat.min(), data_cur_flat.max()), xycoords="data", size=8, va="center", horizontalalignment="center", bbox=dict(boxstyle="round", facecolor="w"))

            ax.legend(loc='lower right', prop={'size':6})

            png_path = startdir + os.sep + "corr_%s_%s.png"%(sf_dir, proc_dir)
            #if not os.path.isfile(png_path):
            plt.savefig(png_path, format='png', dpi=600)
            # Close figure.
            plt.close("all")
            print("Made figure: %s"%png_path)

        # Write intensities
        peaks_results_name = startdir + os.sep + "peaks_%s_results.tab"%(sf_dir)
        write_peak_list(filename=peaks_results_name, x_axis_pts=table['X_AXIS'], y_axis_pts=table['Y_AXIS'], x_axis_ppm=x_axisppm, y_axis_ppm=y_axisppm, int_ref=data_ref_full_int, list_int_cov=proc_ints)

        # Now make report for Hist
        hist_results_name = startdir + os.sep + "hist_%s_results.txt"%(sf_dir)
        hist_results = open(hist_results_name, 'w')

        # Collect header
        headers = ['i', 'data', 'showApod_rmsd', 'hist_sigma', 'hist_amp', 'hist_mu', ]

        # Collect data
        datacsv = []
        datacsv_ref = ["00", '%11s'%'REF', '%4.2f'%res_dic[sf_dir]['ref']['rmsd'], '%4.2f'%res_dic[sf_dir]['ref']['hist'][2], '%4.2f'%res_dic[sf_dir]['ref']['hist'][0], '%4.2f'%res_dic[sf_dir]['ref']['hist'][1]]
        datacsv.append(datacsv_ref)

        for j, proc_dir in enumerate(all_proc_dirs):
            datacsv_cur = ["%02d"%(j+1), '%11s'%proc_dir, '%4.2f'%res_dic[sf_dir][proc_dir]['rmsd'], '%4.2f'%res_dic[sf_dir][proc_dir]['hist'][2], '%4.2f'%res_dic[sf_dir][proc_dir]['hist'][0], '%4.2f'%res_dic[sf_dir][proc_dir]['hist'][1]]
            datacsv.append(datacsv_cur)

        # Write data
        write_data(out=hist_results, headings=headers, data=datacsv)
        hist_results.close()

        # Now make report for correlation
        corr_results_name = startdir + os.sep + "corr_%s_results.txt"%(sf_dir)
        corr_results = open(corr_results_name, 'w')

        # Collect header
        #headers = ['i', 'data', 'a', 'r_xy', 'r_xy^2']
        headers = ['i', 'data', 'a', 'r_xy^2']

        for label, color, sel_mask, dickey, pct in  sn_masks_used:
            headers.append('a_%s'%dickey)
            #headers.append('r_xy_%s'%dickey)
            headers.append('r_xy^2_%s'%dickey)

        # Collect data
        datacsv = []

        for j, proc_dir in enumerate(all_proc_dirs):
            #datacsv_cur = ["%02d"%(j+1), '%11s'%proc_dir, '%3.6f'%res_dic[sf_dir][proc_dir]['corr'][0], '%3.6f'%res_dic[sf_dir][proc_dir]['corr'][1], '%3.6f'%res_dic[sf_dir][proc_dir]['corr'][1]**2]
            datacsv_cur = ["%02d"%(j+1), '%11s'%proc_dir, '%3.6f'%res_dic[sf_dir][proc_dir]['corr'][0], '%3.6f'%res_dic[sf_dir][proc_dir]['corr'][1]**2]

            for label, color, sel_mask, dickey, pct in  sn_masks_used:
                datacsv_cur.append('%3.6f'%res_dic[sf_dir][proc_dir]['corr_s'][dickey][0])
                #datacsv_cur.append('%3.6f'%res_dic[sf_dir][proc_dir]['corr_s'][dickey][1])
                datacsv_cur.append('%3.6f'%res_dic[sf_dir][proc_dir]['corr_s'][dickey][1]**2)

            datacsv.append(datacsv_cur)

        # Write data
        write_data(out=corr_results, headings=headers, data=datacsv)
        corr_results.close()

        # Now make a residual histogram.
        # Since the intensities are modulated by 1.2, we use the full 00_ref.proc as reference.
        data_ref = res_dic[sf_dir]['00_ref.proc']['data']

        # Loop over all proc dirs.
        for j, proc_dir in enumerate(proc_dirs):
            data_cur = res_dic[sf_dir][proc_dir]['data']

            # Data residual
            data_resi = data_cur - data_ref

            # Make a histogram
            ax, amp, mu, sigma, xlim, ylim = hist_plot(ndarray=data_resi, show=False)
            res_dic[sf_dir][proc_dir]['residual_hist'] = [amp, mu, sigma]

            # Set same limits as ref
            #ax.set_xlim(mu-6*sigma, mu+6*sigma)
            ax.set_xlim(-10*rmsd_FULL_ft2file, +10*rmsd_FULL_ft2file)
            #ax.set_ylim(ylim_ref)

            png_path = startdir + os.sep + "residual_hist_%s_%s.png"%(sf_dir, proc_dir)
            if not os.path.isfile(png_path):
                plt.savefig(png_path, format='png', dpi=600)
            # Close figure.
            plt.close("all")
            print("Made figure: %s"%png_path)

        # Now make report for Residual Hist
        hist_results_name = startdir + os.sep + "residual_hist_%s_results.txt"%(sf_dir)
        hist_results = open(hist_results_name, 'w')

        # Collect header
        headers = ['i', 'data', 'hist_sigma', 'hist_amp', 'hist_mu', ]

        # Collect data
        datacsv = []

        for j, proc_dir in enumerate(proc_dirs):
            datacsv_cur = ["%02d"%(j+2), '%11s'%proc_dir, '%4.2f'%res_dic[sf_dir][proc_dir]['residual_hist'][2], '%4.2f'%res_dic[sf_dir][proc_dir]['residual_hist'][0], '%4.2f'%res_dic[sf_dir][proc_dir]['residual_hist'][1]]
            datacsv.append(datacsv_cur)

        # Write data
        write_data(out=hist_results, headings=headers, data=datacsv)
        hist_results.close()

    # Print elapsed time for running script.
    elapsed_time = time.time() - start_time
    print("--- %3.1f s seconds for run time---" % elapsed_time )
