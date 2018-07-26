import os
import getpass
import numpy as np
from ABC_SMC_DVmodel import calibration, MCMC_ABC
from DV_model_sim_along_phy import DVtraitsim_tree

# Observation parameters [gamma,a]
# We would like to explore the behavior of model under
#  gamma = (0,0.001,0.01,0.1,0.5,1)
#  a = (0,0.001,0.01,0.1,0.5,1)
#  K = 100000
par_obs = np.array([0.001,0.1,100000])
scalar = 10000
# Observation generated
# Load the data for a given tree
# The directory of the tree data
if getpass.getuser() == "Hanno":
    dir_path = os.path.dirname(os.path.realpath(__file__))
else:
    # Liang uses hard-coded path
    dir_path = 'C:\\Liang\\Googlebox\\Python\\Project2\\R-tree_sim\\'
file = dir_path + '\\example11\\'
# Simulate data
simresult = DVtraitsim_tree(file = file,replicate = 3, gamma1 = par_obs[0], a = par_obs[1],K = par_obs[2],scalar = scalar)
# We only need the data at tips.
evo_time, total_species = simresult[0].shape
evo_time = evo_time-1
trait_RI_dr = simresult[0]
population_RI_dr = simresult[1]
traitvar = simresult[3]
trait_dr_tips = trait_RI_dr[evo_time,:][~np.isnan(trait_RI_dr[evo_time,:])]
population_tips = population_RI_dr[evo_time,:][~np.isnan(population_RI_dr[evo_time,:])]
traitvar = traitvar[evo_time,:][~np.isnan(traitvar[evo_time,:])]
# observation data
obs = np.array([trait_dr_tips,population_tips,traitvar])


# Calibrication step
cal_size = 10
calidata_file = file+'savedata'
# TEST1: Uniform prior distribution example
priorpar = [0,1,0,1]
collection = calibration(samplesize = cal_size, priorpar = priorpar, treefile = file,calidata_file=calidata_file,
                         K=par_obs[2],scalar = scalar)

# Processing the calibration data. We would like to chose the 5% closest simulation data to the observation
# to provide the prior information for MCMC step.
calidata = np.load(calidata_file + '.npz')
calitrait = calidata['calitrait']
calipop = calidata['calipop']
calivar = calidata['calivar']
calipara = calidata['calipar']

coll = np.empty(shape=(cal_size, 6))
sort_obstrait = np.sort(obs[0])
sort_obsvar = np.sort(obs[2])

for i in range(0,cal_size):
    meandiff_trait = np.linalg.norm(calitrait[i] - obs[0])
    meandiff_trait_sort = np.linalg.norm(np.sort(calitrait[i]) - sort_obstrait)
    meandiff_var = np.linalg.norm(calivar[i] - obs[2])
    meandiff_var_sort = np.linalg.norm(np.sort(calivar[i]) - sort_obsvar)
    coll[i] = np.concatenate((calipara[i],[meandiff_trait],[meandiff_trait_sort],[meandiff_var],[meandiff_var_sort]))


threshold = 0.5
num = threshold*cal_size-1

# Criterion for trait mean distance
# The minimum of the distance of sorted data
delta_mean = np.sort(coll[:,3])[int(num)]
mn_mean,idx_mean = min( (coll[i,3],i) for i in range(len(coll[:,3])) )
# Start value regarding to the minimum distance.
startvalue_mean = coll[idx_mean,:2]
# Filter the calibration data by 5% closest threshold.
filtered_mean = coll[coll[:,3]<=delta_mean]
priorpar_mean = [np.mean(filtered_mean[:,0]),np.std(filtered_mean[:,0]),
                 np.mean(filtered_mean[:,1]),np.std(filtered_mean[:,1])]

# Criterion for trait variance distance
delta_var = np.sort(coll[:,5])[int(num)]
mn_var,idx_var = min( (coll[i,5],i) for i in range(len(coll[:,5])) )
startvalue_var = coll[idx_var,:2]




# ABC_MCMC step
# Chain 1 by criterion of trait mean
iterations = 10
posterior_mean = MCMC_ABC(startvalue= startvalue_mean,K=par_obs[2], iterations = iterations, delta = delta_mean, obs = obs,sort = 1,
                     priorpar=priorpar_mean, file = file, mcmcmode = 'nor',abcmode='mean',scalar = scalar)
file2 = file + 'posterior_mean.txt'
np.savetxt(file2,posterior_mean)

# gap = n: take every nth value.
gap=2
priorpar_var = [np.mean(posterior_mean[::gap,0]),np.std(posterior_mean[::gap,0]),
                np.mean(posterior_mean[::gap,1]),np.std(posterior_mean[::gap,1])]

# Chain 2 by criterion of trait variance
iterations = 10
posterior_var = MCMC_ABC(startvalue= startvalue_var,K=par_obs[2], iterations = iterations, delta = delta_var, obs = obs,sort = 1,
                     priorpar=priorpar_var, file = file, mcmcmode = 'nor',abcmode='variance',scalar = scalar)
file2 = file + 'posterior_var.txt'
np.savetxt(file2,posterior_mean)