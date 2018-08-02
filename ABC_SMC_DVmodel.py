import numpy as np
from DV_model_sim_along_phy import DVtraitsim_tree
import scipy.stats
from scipy.stats import norm

import timeit

def PosNormal(mean, sigma):
    x = np.random.normal(mean,sigma,1)
    return(x if x>=0 else PosNormal(mean,sigma))

def calibration(samplesize, priorpar, treefile,calidata_file,K,scalar,crownage=15):
    collection = np.zeros(shape=(samplesize,2))
    cali_traitdata = ([])
    cali_popdata = ([])
    cali_vardata = ([])
    par_picked = ([])
    total_time = crownage*scalar
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
            #single simulation; to be replaced by C++ simulation.
            sample_cal =  DVtraitsim_tree(file = treefile,scalar=scalar, replicate = 0,K = K, gamma1 = uniform_gamma,a = uniform_a)
            if sample_cal[0]['simtime']< total_time:
                do = 0
            else:
                print('Retry')
                do = 1
        # prossessing tips data; remove nan values.
        trait_RI_dr = sample_cal[0]['Z']
        population_RI_dr = sample_cal[0]['N']
        traitVar = sample_cal[0]['V']
        trait_dr_tips = trait_RI_dr[~np.isnan(trait_RI_dr)]
        population_tips = population_RI_dr[~np.isnan(population_RI_dr)]
        traitVar_tips = traitVar[~np.isnan(traitVar)]
        #collecte complete results and parameters
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
def ABC_acceptance(par,delta,obs,sort, scalar,file,abcmode='mean',crownage=15):
    #Single simulation; to be replaced by C++ simulation.
    sim = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
    total_time = crownage * scalar
    if sim[0]['simtime']< total_time:
        # prossessing tips data; remove nan values.
        trait_RI_dr = sim[0]['Z']
        population_RI_dr = sim[0]['N']
        traitVar = sim[0]['V']
        trait_dr_tips = trait_RI_dr[~np.isnan(trait_RI_dr)]
        population_tips = population_RI_dr[~np.isnan(population_RI_dr)]
        traitVar_tips = traitVar[~np.isnan(traitVar)]
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



# SMC ABC for model selection
def SMC_ABC_MS(timestep, particlesize, obs, epsilon, prior, file,scalar,crownage, K, sort = 0):
    tic = timeit.default_timer()
    d = np.zeros(shape = (timestep, particlesize))  #distance matrix of simulations and obs
    model = np.zeros(shape = (timestep, particlesize))
    gamma = np.zeros(shape = (timestep, particlesize))  # gamma jumps
    a =  np.zeros(shape = (timestep, particlesize))     # a  jumps
    total_simulation = np.zeros(timestep)
    total_time=crownage*scalar
    # prior information [mean_gamma,var_gamma,mean_a,var_a]
    gamma_prior_mean =  prior[0]
    gamma_prior_var = prior[1]
    a_prior_mean = prior[2]
    a_prior_var = prior[3]
    # Initialize thredhold
    epsilon = epsilon
    # Weight vectors for gamma and a
    weight_gamma = np.zeros(shape = (timestep, particlesize))
    weight_gamma.fill(1/particlesize)
    weight_a = np.zeros(shape = (timestep, particlesize))
    weight_a.fill(1/particlesize)
    for t in range(timestep):
        sim_count = 0
        str = 'Time step : %d;' % t
        #Initial round
        if t == 0:
            for i in range(particlesize):
                str_p = str + ' Particle size : %d' % i
                print(str_p)
                d[t,i] = epsilon + 1
                while d[t,i] > epsilon:
                    sim_count += 1
                    # Sample model and parameters uniformly
                    propose_model = np.random.randint(4)
                    propose_gamma = np.random.uniform(0,1,1)
                    propose_a = np.random.uniform(0,1,1)
                    if propose_model == 0:  # BM model
                        # draw parameters from prior information
                        propose_gamma = 0
                        propose_a = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 1:  # Competition model
                        # draw parameters from prior information
                        propose_gamma = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 2: # OU model / Natural selection model
                        # draw parameters from prior information
                        propose_a = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 3: # Natural selection & competition model
                        # draw parameters from prior information
                        par = [propose_gamma,propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])

                    if sample[0]['simtime']<total_time:
                        diff=np.inf
                    else:
                        sampletrait = sample[0]['Z']     # np.array([sample[0], sample[2]])
                        samplearray = sampletrait[~np.isnan(sampletrait)]
                        obstrait = obs[0]['Z']                               # np.array([obs[0], obs[2]])
                        obsarray = obstrait[~np.isnan(obstrait)]
                        # calculate the distance between simulation and obs
                        if sort == 0:
                            diff = np.linalg.norm(samplearray - obsarray)
                        else:
                            samplearray_sort = samplearray[:, samplearray[0, :].argsort()]
                            obsarray_sort = obsarray[:, obsarray[0, :].argsort()]
                            diff = np.linalg.norm(samplearray_sort - obsarray_sort)
                    d[t, i] = diff
                # record the accepted values
                gamma[t, i] = propose_gamma
                a[t,i] = propose_a
                model[t,i] = propose_model
        else:
            # shrink the threshold by 40 percentile for each time step
            epsilon = np.append(epsilon, np.percentile(d[t-1,],40))
            # calculate weighted variance of the parameters at previous time step
            gamma_pre_mean = np.sum(gamma[t-1,] * weight_gamma[t-1,])
            gamma_pre_var = np.sum(( gamma[t-1,] - gamma_pre_mean)**2 * weight_gamma[t-1,])
            a_pre_mean = np.sum(a[t - 1,] * weight_a[t - 1,])
            a_pre_var = np.sum((a[t - 1,] - a_pre_mean) ** 2 * weight_a[t - 1,])
            for i in range(particlesize):
                str_p = str + ' Particle size : %d' % i
                print(str_p)
                d[t, i] = epsilon[t] + 1
                while d[t,i] > epsilon[t]:
                    sim_count += 1
                    # Sample model
                    propose_model = np.random.randint(4)
                    # sample the parameters by the weight
                    sample_gamma_index = np.random.choice(particlesize,1, p = weight_gamma[t-1,])
                    sample_a_index = np.random.choice(particlesize,1, p = weight_a[t-1,])
                    # mean of the sample for gamma
                    propose_gamma0 = gamma[t-1,sample_gamma_index-1]
                    # draw new gamma with mean and variance
                    propose_gamma = abs(np.random.normal(propose_gamma0,np.sqrt(2*gamma_pre_var)))
                    # mean of the sample for a
                    propose_a0 = a[t-1,sample_a_index-1]
                    # draw new a with mean and variance
                    propose_a = abs(np.random.normal(propose_a0,np.sqrt(2* a_pre_var)))
                    if propose_model == 0:  # BM model
                        # draw parameters from prior information
                        propose_gamma = 0
                        propose_a = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 1:  # Competition model
                        # draw parameters from prior information
                        propose_gamma = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 2: # OU model / Natural selection model
                        # draw parameters from prior information
                        propose_a = 0
                        par = [propose_gamma, propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])
                    elif propose_model == 3: # Natural selection & competition model
                        # draw parameters from prior information
                        par = [propose_gamma,propose_a,K]
                        # simulate under the parameters
                        sample = DVtraitsim_tree(file=file, scalar=scalar, replicate=0, gamma1=par[0], a=par[1], K=par[2])

                    if sample[0]['simtime']<total_time:
                        diff=np.inf
                    else:
                        sampletrait = sample[0]['Z']     # np.array([sample[0], sample[2]])
                        samplearray = sampletrait[~np.isnan(sampletrait)]
                        obstrait = obs[0]['Z']                               # np.array([obs[0], obs[2]])
                        obsarray = obstrait[~np.isnan(obstrait)]
                        # calculate the distance between simulation and obs
                        if sort == 0:
                            diff = np.linalg.norm(samplearray - obsarray)
                        else:
                            samplearray_sort = samplearray[:, samplearray[0, :].argsort()]
                            obsarray_sort = obsarray[:, obsarray[0, :].argsort()]
                            diff = np.linalg.norm(samplearray_sort - obsarray_sort)
                    d[t, i] = diff
                gamma[t, i] = propose_gamma
                a[t,i] = propose_a
                model[t,i] = propose_model
                # compute new weights for gamma and a
                weight_gamma_denominator = np.sum(weight_gamma[t-1,]* norm.pdf(propose_gamma,gamma[t-1,] ,
                                                                               np.sqrt(2*gamma_pre_var)))
                weight_gamma_numerator = norm.pdf(propose_gamma,gamma_prior_mean,gamma_prior_var)
                weight_gamma[t,i] = weight_gamma_numerator / weight_gamma_denominator

                weight_a_denominator = np.sum(weight_a[t - 1,] * norm.pdf(propose_a, a[t - 1,],
                                                                                  np.sqrt(2 * a_pre_var)))
                weight_a_numerator = norm.pdf(propose_a, a_prior_mean, a_prior_var)
                weight_a[t, i] = weight_a_numerator / weight_a_denominator
        # normalize the weights
        total_simulation[t] = sim_count
        weight_gamma[t,] = weight_gamma[t,]/sum(weight_gamma[t,])
        weight_a[t,] = weight_a[t,]/sum(weight_a[t,])
    # create the dictionary for output
    SMC_ABC_model = {'gamma': gamma, 'a': a, 'model': model, 'weight_gamma':weight_gamma,'weight_a':weight_a,'error':epsilon,'diff':d,
               'tot_sim':total_simulation}
    toc = timeit.default_timer()
    elapse = toc - tic
    timetext = 'Elapsed time: %.2f' % elapse
    print(timetext)
    return SMC_ABC_model
