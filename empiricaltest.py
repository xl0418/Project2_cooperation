import os
import sys
import platform
sys.path.append('C:/Liang/abcpp_master/abcpp')
from dvtraitsim_shared import DVTreeData, DVParam
import dvtraitsim_cpp as dvcpp

full =1

#full tree and pruned tree directory
dir_path = 'c:/Liang/Googlebox/Research/Project2/planktonic_foraminifera_macroperforate/'
if full==1:
    files = dir_path + 'full_data/'
else:
    files = dir_path + 'pruend_data/'

td = DVTreeData(path=files, scalar=100)
K=10e8
nu=1/(100*K)

obs_param = DVParam(gamma=0.01, a=0.5, K=K, nu=nu, r=1, theta=0, Vmax=1, inittrait=0, initpop=500,
             initpop_sigma = 10.0, break_on_mu=False)

pop = dvcpp.DVSim(td, obs_param)