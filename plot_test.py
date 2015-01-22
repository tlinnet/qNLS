import qNLS
import matplotlib.pyplot as plt
import nmrglue
import numpy
import copy

dic_ref, udic_ref, data_ref = qNLS.read_spectrum(file='test.ft2')

table = nmrglue.analysis.peakpick.pick(data=data_ref, pthres=100000., nthres=None, algorithm='connected', est_params=False, cluster=False, table=True)

ax = qNLS.contour_plot(dic=dic_ref, udic=udic_ref, data=data_ref, contour_start=30000., contour_num=10, contour_factor=1.20, ppm=True, show=False, table=table)


dic_resi, udic_resi, data_resi = qNLS.read_spectrum(file='test_resi.ft2')
ax = qNLS.contour_plot(dic=dic_resi, udic=udic_resi, data=data_resi, contour_start=5000., contour_num=5, contour_factor=1.20, ppm=True, show=False, table=table)

print table['X_AXIS']
print table['Y_AXIS']

print data_ref.shape

print table['X_AXIS'][0], table['Y_AXIS'][0]
print data_ref[table['Y_AXIS'][0], table['X_AXIS'][0]]
print data_ref[table['Y_AXIS'][-1], table['X_AXIS'][-1]]
#print data_ref[27, 40]
int_arr = data_ref[table['Y_AXIS'].astype(int), table['X_AXIS'].astype(int)]
int_arr2 = copy.deepcopy(int_arr) * 1.2

plt.close("all")
#plt.show()

contour_start=30000.
contour_num=20
contour_factor=1.20
contour_start * contour_factor ** numpy.arange(contour_num)
cl_pos = contour_start * contour_factor ** numpy.arange(contour_num)
plt.contour(data_ref, cl_pos)

plt.plot(table['X_AXIS'], table['Y_AXIS'], 'gx')

plt.show()
