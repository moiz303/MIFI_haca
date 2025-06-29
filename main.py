from make_hash import Hash
from scipy.optimize import nnls

res = Hash('file.jpg').get_grey()
print(res)