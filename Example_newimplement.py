import sys, os
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Liang/Code/Pro2/Python_p2')
elif platform.system()=='Darwin':
    sys.path.append('/Users/dudupig/Documents/GitHub/Code/Pro2/Python_p2')
from DVmodelsim import DVtraitsim_tree
if platform.system()=='Windows':
    file = 'C:\\Liang\\Code\\Pro2\\abcpp\\tree_data\\example1\\'
elif platform.system()=='Darwin':
    file = '/Users/dudupig/Documents/GitHub/Code/Pro2/abcpp/tree_data/example3/'
gamma1=0.001
a=0.01
scalor=10000
K=10e8
nu=1/(100*K)

simresult = DVtraitsim_tree(file=file, gamma1=gamma1, a=a, K=K, scalar=scalor,nu=nu,Vmax=1)
simresult[2]
