import numpy as np
from DV_model_sim_along_phy import DVtraitsim_tree
import scipy.stats
import timeit

def PosNormal(mean, sigma):
    x = np.random.normal(mean,sigma,1)
    return(x if x>=0 else PosNormal(mean,sigma))

def calibration(samplesize, priorpar, treefile,calidata_file,K,scalar):
    collection = np.zeros(shape=(samplesize,2))
    cali_traitdata = ([])
    cali_popdata = ([])
    cali_vardata = ([])
    par_picked = ([])
    for i in range(samplesize):
        do = 0
        while(do==0):
            uniform_gamma = np.random.uniform(priorpar[0], priorpar[1], 1)
            uniform_a = np.random.uniform(priorpar[2], priorpar[3], 1)
            print(i)
            par_cal = np.zeros(2)
            par_cal[0] = uniform_gamma
            par_cal[1] = uniform_a
            par_picked.append(par_cal)
            sample_cal =  DVtraitsim_tree(file = treefile,scalar=scalar, replicate = 0,K = K, gamma1 = uniform_gamma,a = uniform_a)
            if sample_cal[2]:
                do = 1
            else:
                print('Retry')
                do = 0
        trait_RI_dr = sample_cal[0]
        population_RI_dr = sample_cal[1]
        traitVar = sample_cal[3]
        evo_time, total_species = sample_cal[0].shape
        evo_time = evo_time - 1
        trait_dr_tips = trait_RI_dr[evo_time, :][~np.isnan(trait_RI_dr[evo_time, :])]
        population_tips = population_RI_dr[evo_time, :][~np.isnan(population_RI_dr[evo_time, :])]
        traitVar_tips = traitVar[evo_time, :][~np.isnan(traitVar[evo_time, :])]
        collection[i] = np.array(par_cal)
        cali_traitdata.append(trait_dr_tips)
        cali_popdata.append(population_tips)
        cali_vardata.append(traitVar_tips)
    cali_traitdataarray = np.asarray(cali_traitdata)
    cali_popdataarray = np.asarray(cali_popdata)
    cali_vardataarray = np.asarray(cali_vardata)
    calipar=collection[:,:2]
    par_pickedarray = np.asarray(par_picked)
    np.savez(calidata_file,calipar = calipar, calitrait=cali_traitdataarray, calipop =cali_popdataarray, calivar=cali_vardataarray,picked = par_pickedarray)
    return collection


# par = (gamma1, a, K)
def ABC_acceptance(par,delta,obs,sort, scalar,file,abcmode='mean'):
    sim = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
    if sim[2]:
        trait_RI_dr = sim[0]
        population_RI_dr = sim[1]
        traitVar = sim[3]
        evo_time, total_species = sim[0].shape
        evo_time = evo_time - 1
        trait_dr_tips = trait_RI_dr[evo_time, :][~np.isnan(trait_RI_dr[evo_time, :])]
        population_tips = population_RI_dr[evo_time, :][~np.isnan(population_RI_dr[evo_time, :])]
        traitVar_tips = traitVar[evo_time, :][~np.isnan(traitVar[evo_time, :])]
        sample = np.array([trait_dr_tips, population_tips, traitVar_tips])

        if sort == 0:
            if abcmode == 'mean':
                diff = np.linalg.norm(sample[0] - obs[0])
            elif abcmode == 'variance':
                diff = np.linalg.norm(sample[2] - obs[2])
            else:
                print('Please indicate mode')
                diff = np.inf
            if diff < delta:
                return True
            else:
                return False
        else:
            if abcmode == 'mean':
                diff_sort = np.linalg.norm(np.sort(sample[0]) - np.sort(obs[0]))
            elif abcmode == 'variance':
                diff_sort = np.linalg.norm(np.sort(sample[2]) - np.sort(obs[2]))
            else:
                print('Please indicate mode')
                diff_sort = np.inf
            if diff_sort < delta:
                return True
            else:
                return False
    else:
        return False





def MCMC_ABC(startvalue, iterations,delta,obs,sort,priorpar, file,K,scalar, mcmcmode = 'uni',abcmode='mean'):
    tic = timeit.default_timer()
    MCMC = np.zeros(shape=(iterations+1,2))
    MCMC[0,] = startvalue
    par_jump = np.empty(3)
    par_jump[2] = K
    if mcmcmode == 'uni':
        for i in range(iterations):
            par_jump[0] = np.random.uniform(priorpar[0],priorpar[1])
            par_jump[1] = np.random.uniform(priorpar[2],priorpar[3])

            if (ABC_acceptance(par = par_jump,delta = delta, obs = obs,sort = sort, file = file,abcmode=abcmode,scalar=scalar)):
                MCMC[i+1,] = par_jump[:2]
                print("MCMC : %d Accepted" % (i+1))

            else:
                MCMC[i + 1,] = MCMC[i ,]
                print("MCMC : %d Rejected" % (i+1))
    elif mcmcmode == 'nor':
        for i in range(iterations):
            par_jump[0] = abs(np.random.normal(loc=MCMC[i, 0], scale=0.01))
            par_jump[1] = abs(np.random.normal(loc=MCMC[i, 1], scale=0.01))

            pro = np.random.uniform(0,1,1)[0]
            pro_gamma1 = scipy.stats.norm(priorpar[0], priorpar[1]).pdf(par_jump[0])
            pro_gamma2 = scipy.stats.norm(priorpar[0], priorpar[1]).pdf(MCMC[i ,0])
            pro_a1 = scipy.stats.norm(priorpar[2], priorpar[3]).pdf(par_jump[1])
            pro_a2 = scipy.stats.norm(priorpar[2], priorpar[3]).pdf(MCMC[i ,1])

            pro_ratio = (pro_gamma1*pro_a1)/(pro_gamma2*pro_a2)
            accept_criterion = np.min([1,pro_ratio])
            if ABC_acceptance(par = par_jump, delta=delta, obs=obs, sort=sort, file = file,abcmode=abcmode,
                     scalar=scalar) and (pro <= accept_criterion):
                MCMC[i + 1,] = par_jump[:2]
                print("MCMC : %d Accepted" % (i + 1))
            else:
                MCMC[i + 1,] = MCMC[i,]
                print("MCMC : %d Rejected" % (i + 1))
    toc = timeit.default_timer()
    elapse = toc - tic
    timetext = 'Elapsed time: %.2f' % elapse
    print(timetext)
    return MCMC
