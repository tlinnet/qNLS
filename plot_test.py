import qNLS
import matplotlib.pyplot as plt
import nmrglue
import numpy
import copy
import sys

dic_ref, udic_ref, data_ref = qNLS.read_spectrum(file='test.ft2')

table = nmrglue.analysis.peakpick.pick(data=data_ref, pthres=100000., nthres=None, algorithm='connected', est_params=False, cluster=False, table=True)
uc_dim0 = nmrglue.pipe.make_uc(dic_ref, data_ref, dim=0)
uc_dim1 = nmrglue.pipe.make_uc(dic_ref, data_ref, dim=1)

y_axisppm = uc_dim0.unit(table['Y_AXIS'], "ppm")
x_axisppm = uc_dim1.unit(table['X_AXIS'], "ppm")

ax = qNLS.contour_plot(dic=dic_ref, udic=udic_ref, data=data_ref, contour_start=30000., contour_num=10, contour_factor=1.20, ppm=True, show=False, table=table)

dic_resi, udic_resi, data_resi = qNLS.read_spectrum(file='test_resi.ft2')
ax = qNLS.contour_plot(dic=dic_resi, udic=udic_resi, data=data_resi, contour_start=5000., contour_num=5, contour_factor=1.20, ppm=True, show=False, table=table)

int_arr = data_ref[table['Y_AXIS'].astype(int), table['X_AXIS'].astype(int)]

plt.close("all")
#plt.show()

dic_resi, udic_resi, data_resi = qNLS.read_spectrum(file='test_resi.ft2')

int_arr_resi = data_resi[table['Y_AXIS'].astype(int), table['X_AXIS'].astype(int)] + int_arr

qNLS.write_peak_list(filename="test.tab", x_axis_pts=table['X_AXIS'], y_axis_pts=table['Y_AXIS'], x_axis_ppm=x_axisppm, y_axis_ppm=y_axisppm, int_ref=int_arr, list_int_cov=[int_arr, int_arr_resi])

