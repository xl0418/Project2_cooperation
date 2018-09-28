from operator import itemgetter, attrgetter
from matplotlib.pylab import *
import numpy as np


class DVTreeData:
    def __init__(self, path, scalar):
        self.path = path
        # load tree data
        self.timelist = self._from_txt(path + 'timelist.csv')
        self.timebranch = self._from_txt(path + 'timebranch.csv')
        self.timeend = self._from_txt(path + 'timeend.csv')
        self.traittable = self._from_txt(path + 'traittable.csv')
        self.ltable = self._from_txt(path + 'Ltable.csv')
        # derived data
        self.parent_index = np.absolute(self.ltable[:, 1]).astype(int)
        self.daughter_index = np.absolute(self.ltable[:, 2]).astype(int)
        self.evo_timelist = (scalar * (max(self.timelist[:, 0]) - self.timelist[:, 0])).astype(int)
        self.timebranch = self.timebranch[:, 0].astype(int) - 1
        self.timeend = self.timeend[:, 0].astype(int) - 1
        # evolution time: speciation time
        self.evo_time = max(self.evo_timelist)
        self.speciate_time = self.evo_timelist[self.timebranch]
        self.extinct_time = self.evo_timelist[self.timeend]
        self.extinct_time[self.extinct_time == self.evo_time] = -1
        self.extinct_time = self.extinct_time[self.extinct_time < self.evo_time]
        self.total_species = len(self.speciate_time)
        # sanity check
        event_times = np.append(self.speciate_time[1:,], self.extinct_time[self.extinct_time != -1]) # omit first '0' in speciation times
        if len(event_times) != len(np.unique(event_times)):
            print('Scalar is too small that some adjunct events are not distinguishabel')
        # create event list: [time, parent, daughter]
        # extinction event if daughter == -1, speciation event otherwise
        self.events = sorted(self._speciation_events() + self._extinction_events())
        self.events.append([-1,-1,-1])  # guard

    # returns trimmed table as numpy.ndarray
    def _from_txt(self, file):
        tmp = np.genfromtxt(file, delimiter=',', skip_header=1)
        return np.delete(tmp, (0), axis=1)

    # creates list of speciation events [time, parent, daughter]
    def _speciation_events(self):
        speciation_events = list()
        for sp in range(2, len(self.speciate_time)):
            speciation_events.append([self.speciate_time[sp], self.parent_index[sp] - 1, self.daughter_index[sp]- 1])
        return speciation_events

    # creates list of extinction events [time, specie, -1]
    def _extinction_events(self):
        extinction_events = list()
        for se in range(0, len(self.extinct_time)):
            time = self.extinct_time[se]
            if time != -1:
                extinction_events.append([time, np.where(self.extinct_time == time)[0][0], -1])
        return extinction_events


# competition functions
# returns beta = Sum_j( exp(-a(zi-zj)^2) * Nj)
#         sigma = Sum_j( 2a * (zi-zj) * exp(-a(zi-zj)^2) * Nj)
#         sigmaSqr = Sum_j( 4a^2 * (zi-zj)^2 * exp(-a(zi-zj)^2) * Nj)
def competition_functions(a, zi, nj):
    T = zi[:, np.newaxis] - zi  # trait-distance matrix (via 'broadcasting')
    t1 = np.exp(-a * T ** 2) * nj
    t2 = (2 * a) * T
    beta = np.sum(t1, axis=1)
    sigma = np.sum(t2 * t1, axis=1)
    sigmasqr = np.sum(t2 ** 2 * t1, axis=1)
    return beta, sigma, sigmasqr

# Sample function within a specific range (0,1)
def PopsplitNormal(mean, sigma):
    x = np.random.normal(mean,sigma,1)
    return(x if x>0 and x<1 else PopsplitNormal(mean,sigma))

def DVtraitsim_tree(file, gamma1, a, K, scalar, nu=0.00000001,keep_alive=1, r=1,theta=0, Vmax=1, replicate=0,initrait=0,inipop=500):
    valid = True
    if replicate > 0:
        np.random.seed(replicate)  # set random seed
    td = DVTreeData(file, scalar)

    # Initialize trait evolution and population evolution matrices
    trait_RI_dr = np.zeros((td.evo_time + 1, td.total_species))  # trait
    population_RI_dr = np.zeros((td.evo_time + 1, td.total_species))  # population
    V = np.zeros((td.evo_time + 1, td.total_species))  # trait vairance

    #  initialize condition for species trait and population
    trait_RI_dr[0, (0, 1)] = initrait  # trait for species
    mu_pop, sigma_pop = inipop, 10  # mean and standard deviation
    population_RI_dr[0, (0, 1)] = np.random.normal(mu_pop, sigma_pop, 2)
    V[0] = (1 / td.total_species)
    # pull event list
    events = td.events.copy()
    next_event = events.pop(0)   # remove the first item of the list, e.g. events, and give the removed iterm to next_event
    # existing species matrix
    existing_species = td.traittable
    node = 0
    idx = np.where(existing_species[node] == 1)[0]    # existing species
    # trait-population coevolution model
    for i in range(td.evo_time):
        # pull current state
        Ni = population_RI_dr[i, idx]
        Vi = V[i, idx]
        # Vmax = np.max(Vi)
        zi = trait_RI_dr[i, idx]
        Ki = K
        dtz = theta - zi
        beta, sigma, sigmasqr = competition_functions(a=a, zi=zi, nj=Ni)

        # update
        var_trait = Vi / (2 * Ni)
        trait_RI_dr[i + 1, idx] = zi + Vi * (2 * gamma1 * dtz + 1 / Ki * sigma) + np.random.normal(0, var_trait, len(idx))
        possion_lambda = Ni * r * np.exp(-gamma1 * dtz**2 + (1 - beta / Ki))
        population_RI_dr[i + 1, idx] = np.maximum(np.random.poisson(lam=possion_lambda),keep_alive)  #, size=(1, len(idx))
        V[i + 1, idx] = Vi / 2 + 2 * Ni * nu * Vmax / (1 + 4 * Ni * nu) \
                        + Vi ** 2 * (
                            -2 * gamma1 + 4 * gamma1**2 * dtz ** 2 +
                                1 / Ki * (2*a*beta - sigmasqr) + 4 * gamma1 / Ki *
                                dtz * sigma + sigma ** 2 / Ki**2
                            )
        # sanity check
        if np.any(population_RI_dr[i + 1, idx] < 1):
            valid = False
            print('Inconsistent zero population')
            break
        if np.any(V[i + 1, idx] < 0):
            valid = False
            print('Negative variance')
            break
        if np.any(V[i + 1, idx] > 100000):
            valid = False
            print('variance>100000 ')
            break
        # events
        while (i+1) == next_event[0]:
        # if (i + 1) == next_event[0]:
            parent = next_event[1]
            daughter = next_event[2]
            if (daughter == -1):
                # extinction
                extinct_species = next_event[1]
                trait_RI_dr[i + 1, extinct_species] = None
                population_RI_dr[i + 1, extinct_species] = 0
            else:
                # speciation
                splitratio = PopsplitNormal(mean=0.5, sigma=0.2)
                trait_RI_dr[i + 1, daughter] = trait_RI_dr[i + 1, parent]
                tmp_pop = population_RI_dr[i + 1, parent]
                population_RI_dr[i + 1, parent] = splitratio * tmp_pop
                population_RI_dr[i + 1, daughter] = (1 - splitratio) * tmp_pop
                V[i + 1, parent] = 1 / 2 * V[i + 1, parent]
                V[i + 1, daughter] = V[i + 1, parent]
            # advance to next event/node
            next_event = events.pop(0)
            node = node + 1
            idx = np.where(existing_species[node] == 1)[0]

    row_ext = np.where(population_RI_dr == 0)[0]
    col_ext = np.where(population_RI_dr == 0)[1]
    trait_RI_dr[row_ext, col_ext] = None
    population_RI_dr[row_ext, col_ext] = None
    V[row_ext, col_ext] = None
    return trait_RI_dr, population_RI_dr, valid, V

