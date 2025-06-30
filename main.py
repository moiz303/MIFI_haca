from make_hash import Hash, res_color
from scipy.optimize import nnls
import matplotlib

first = Hash('templates/file.jpg').get_grey()
second = Hash('templates/file2.jpg').get_grey()
third = Hash('templates/file3.jpg').get_grey()
res = Hash('templates/res.jpg').get_grey()
res_color(first, 'fir.png')
res_color(second, 'sec.png')
res_color(third, 'thi.png')
res_color(res, 'res.png')
