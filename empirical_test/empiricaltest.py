import os
import sys
import platform
sys.path.append('C:/Liang/abcpp_emp/abcpp')
from dvtraitsim_shared import DVTreeData, DVParam
import dvtraitsim_cpp as dvcpp
from dvtraitsim_py import DVSim

full =1

#full tree and pruned tree directory
dir_path = 'c:/Liang/Googlebox/Research/Project2/planktonic_foraminifera_macroperforate/'
if full==1:
    files = dir_path + 'full_data/'
else:
    files = dir_path + 'pruend_data/'

td = DVTreeData(path=files, scalar=1000)
K=10e8
nu=1/(100*K)

obs_param = DVParam(gamma=0.0001, a=0.01, K=K, nu=nu, r=1, theta=0, Vmax=1, inittrait=0, initpop=500000,
             initpop_sigma = 10.0, break_on_mu=False)

for i in range(100):
    print(i)
    # pop = dvcpp.DVSim(td, obs_param)
    pop = DVSim(td,obs_param)
    print(pop['sim_time'])
    if pop['sim_time'] == td.sim_evo_time:
        break