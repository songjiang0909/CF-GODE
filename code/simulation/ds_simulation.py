import numpy as np
from scipy.stats import truncnorm
from simulation.simulation_utils import *
from sklearn.decomposition import LatentDirichletAllocation


def sigmod(x):
    return 1/(1 + np.exp(-x))


def data_simulation(args,dataset,norm):

    print ("simulation starts...")
    data,parts = read_data(dataset)
    train_index,val_index,test_index = data_split(parts)
    train_adj, val_adj, test_adj = adj_split(data,train_index,val_index,test_index,dataset)
    train_x, val_x, test_x = covariate_transform(data,args.dimension,train_index,val_index,test_index)
    w_z1 = 2 * np.random.random_sample((args.dimension)) - 1
    w_z2 = 2 * np.random.random_sample((args.dimension)) - 1

    print ("Now train simulation")
    train_outputs = simulation(args,train_adj,train_x,w_z1,w_z2,norm)
    print ("Now val simulation")
    val_outputs = simulation(args,val_adj,val_x,w_z1,w_z2,norm)
    print ("Now test simulation")
    test_outputs = simulation(args,test_adj,test_x,w_z1,w_z2,norm)

    return train_outputs, val_outputs, test_outputs



def covariate_transform(data,dimension,trainIndex,valIndex,testIndex):

    X = data["Attributes"]
    lda = LatentDirichletAllocation(n_components=dimension)
    lda.fit(X)
    X = lda.transform(X)
    trainX = X[trainIndex]
    valX = X[valIndex]
    testX = X[testIndex]
    print ("Shape of graph covariate train:{}, val:{}, test:{}".format(trainX.shape,valX.shape,testX.shape))

    return trainX,valX,testX



def simulation(args,A,X,w_z1,w_z2,norm):

    '''
    Code adapted from https://github.com/seedatnabeel/TE-CDE
    '''
    
    # gamma_a = args.gamma_a
    # gamma_n = args.gamma_n
    # gamma_f = args.gamma_f
    gamma_a = args.gamma_a
    gamma_n = args.gamma_a
    gamma_f = args.gamma_a
    
    gamma_nf = gamma_f/args.per_neighbor
    print (f"gamma_a :{gamma_a}, gamma_n :{gamma_n}, gamma_f :{gamma_f}, gamma_nf :{gamma_nf}")

    coeff_y = 0.001
    coeff_ny = 0.001/args.per_neighbor
    delta_a = 5
    delta_n = 5
    num_time_steps = args.num_time_steps
    window_size = 5
    treatment_dosage = 1
    drug_half_life = 1

    global_min = 0.1
    global_max = 10


    K = 10
    rho = -0.001
    beta_c = 0.03
    beta_n = beta_c/args.per_neighbor


    ratio = args.flip_rate
    print ("ratio:{}".format(ratio))

    num_units = A.shape[0]
    adj = A
    feature = X


    effect_f2t = np.matmul(w_z1,X.T)
    temp = np.matmul(adj,effect_f2t)
    nei_sum = np.sum(adj,1)
    avg_neighbor_effect_f2t = np.divide(temp,nei_sum)

    effect_f2y = np.matmul(w_z2,X.T)
    temp = np.matmul(adj,effect_f2y)
    nei_sum = np.sum(adj,1)
    avg_neighbor_effect_f2y = np.divide(temp,nei_sum)

    print ("mean :{}".format(np.mean(effect_f2y)))
    print ("avg_neighbor_effect_f2y :{}".format(np.mean(avg_neighbor_effect_f2y)))

    # Patient health stage. (mu, sigma, lower bound, upper bound) 
    health_stages_stats = {
        "I": (1.5, 1, global_min, 3),
        "II": (4.5, 1, 3, 6),
        "III": (8, 1, 6, global_max),
    }

    
    possible_stages = list(health_stages_stats.keys())
    possible_stages.sort()
    initial_stages = np.random.choice(possible_stages,num_units)

    print ("Initial simulation")
    initial_health = []
    for i in range(num_units):
        stg = initial_stages[i]
        mu, sigma, lower_bound, upper_bound = health_stages_stats[stg]
        initial_health_by_stage = np.random.normal(mu, sigma, 1)[0]
        if initial_health_by_stage<lower_bound:
            initial_health_by_stage = lower_bound
        elif initial_health_by_stage>upper_bound:
            initial_health_by_stage = upper_bound
        initial_health.append(initial_health_by_stage)

    #simulation initial health condition
    health_condition = np.zeros((num_units, num_time_steps))
    health_condition[:,0] = initial_health

    cf_health_condition = np.zeros((num_units, num_time_steps))
    cf_health_condition[:,0] = initial_health


    print ("Factual simulation...")
    #simulation
    treatment_probs = np.zeros((num_units, num_time_steps))
    treatment_application_rvs = np.random.rand(num_units, num_time_steps)

    action_application_point = np.zeros((num_units, num_time_steps))
    avg_neighbor_action_application = np.zeros((num_units, num_time_steps))
    action_dosage = np.zeros((num_units, num_time_steps))
    avg_neighbor_action_dosage = np.zeros((num_units, num_time_steps))

    avg_neighbor_outcome = np.zeros((num_units, num_time_steps))
    cf_avg_neighbor_outcome = np.zeros((num_units, num_time_steps))

    avg_health_used_for_all = np.zeros((num_units, num_time_steps))
    avg_neighbor_health_used_for_all = np.zeros((num_units, num_time_steps))

    noise_terms = 0.01 * np.random.randn(num_units,num_time_steps) 

    for t in range(0, num_time_steps - 1):
        for i in range(num_units):
            health_used = health_condition[i, max(t - window_size, 0) : t + 1]
            avg_health_used_for_all[i][t] = np.array([health for health in health_used]).mean()
        temp = np.matmul(adj,avg_health_used_for_all[:,t])
        nei_sum = np.sum(adj,1)
        avg_neighbor_health_t = np.divide(temp,nei_sum)
        avg_neighbor_health_used_for_all[:,t] = avg_neighbor_health_t

        for i in range(num_units):
            current_dose = 0.0
            previous_dose = 0.0 if t == 0 else action_dosage[i, t - 1]

            avg_health_used = avg_health_used_for_all[i][t]
            avg_neighbor_health_used = avg_neighbor_health_used_for_all[i][t]

            treatment_prob_t = sigmod(-1*(gamma_a*(avg_health_used-delta_a)+gamma_n*(avg_neighbor_health_used-delta_n))+gamma_f*effect_f2t[i]+gamma_nf*avg_neighbor_effect_f2t[i])
            treatment_probs[i, t] = treatment_prob_t

            if treatment_application_rvs[i, t] < treatment_prob_t:
                # pass
                action_application_point[i, t] = 1
                current_dose = treatment_dosage


            # Update chemo dosage
            action_dosage[i, t] = previous_dose * np.exp(-np.log(2) / drug_half_life)+ current_dose

        
        temp = np.matmul(adj,action_dosage[:,t])
        nei_sum = np.sum(adj,1)
        avg_neighbor_dosage = np.divide(temp,nei_sum)
        avg_neighbor_action_dosage[:,t] = avg_neighbor_dosage


        temp = np.matmul(adj,action_application_point[:,t])
        nei_sum = np.sum(adj,1)
        avg_neighbor_treatment = np.divide(temp,nei_sum)
        avg_neighbor_action_application[:,t] = avg_neighbor_treatment

        temp = np.matmul(adj,health_condition[:,t])
        nei_sum = np.sum(adj,1)
        avg_neighbor_health = np.divide(temp,nei_sum)
        avg_neighbor_outcome[:,t] = avg_neighbor_health
  

        for i in range(num_units):

            health_condition[i, t + 1] = health_condition[i, t] * (
                1
                +rho * np.log10(K / health_condition[i, t])
                +(rho/args.per_neighbor) * np.log10(K / avg_neighbor_outcome[i, t])
                +coeff_y*effect_f2y[i]+avg_neighbor_effect_f2y[i]*coeff_ny
                + beta_c * action_dosage[i, t]
                + (beta_n * avg_neighbor_action_dosage[i, t])
                + noise_terms[i][t]
            ) 

            if  health_condition[i, t + 1] > global_max:
                health_condition[i, t + 1] = global_max
            if  health_condition[i, t + 1] < global_min:
                health_condition[i, t + 1] = global_min


    print ("Counterfactual simulation...")
    cf_action_application_point = np.zeros((num_units, num_time_steps))
    cf_avg_neighbor_action_application = np.zeros((num_units, num_time_steps))


    r,c = action_application_point.shape
    flip_mask = 1.0*(np.random.rand(r,c)<=ratio)
    for ii in range(r):
        for jj in range(c):
            if flip_mask[ii][jj]==1:
                cf_action_application_point[ii][jj] = action_application_point[ii][jj]*-1+1
            else:
                cf_action_application_point[ii][jj] = action_application_point[ii][jj]


    cf_action_dosage = np.zeros((num_units, num_time_steps))
    cf_avg_neighbor_action_dosage = np.zeros((num_units, num_time_steps))

    cf_noise_terms = 0.01 * np.random.randn(num_units,num_time_steps) 


    for t in range(0, num_time_steps - 1):
        for i in range(num_units):
            current_dose = 0.0
            previous_dose = 0.0 if t == 0 else cf_action_dosage[i, t - 1]

            if cf_action_application_point[i, t] ==1:
                current_dose = treatment_dosage

            # Update chemo dosage
            cf_action_dosage[i, t] = previous_dose * np.exp(-np.log(2) / drug_half_life)+ current_dose


        temp = np.matmul(adj,cf_action_dosage[:,t])
        nei_sum = np.sum(adj,1)
        cf_avg_neighbor_dosage = np.divide(temp,nei_sum)
        cf_avg_neighbor_action_dosage[:,t] = cf_avg_neighbor_dosage

        temp = np.matmul(adj,cf_action_application_point[:,t])
        nei_sum = np.sum(adj,1)
        cf_avg_neighbor_treatment = np.divide(temp,nei_sum)
        cf_avg_neighbor_action_application[:,t] = cf_avg_neighbor_treatment

        temp = np.matmul(adj,cf_health_condition[:,t])
        nei_sum = np.sum(adj,1)
        cf_avg_neighbor_health = np.divide(temp,nei_sum)
        cf_avg_neighbor_outcome[:,t] = cf_avg_neighbor_health

        for i in range(num_units):

            cf_health_condition[i, t + 1] = cf_health_condition[i, t] * (
                1
                +rho * np.log10(K / cf_health_condition[i, t])
                +(rho/3.0) * np.log10(K / cf_avg_neighbor_outcome[i, t])
                +coeff_y*effect_f2y[i]+avg_neighbor_effect_f2y[i]*coeff_ny
                + beta_c * cf_action_dosage[i, t]
                + (beta_n * cf_avg_neighbor_action_dosage[i, t])
                + cf_noise_terms[i][t]
            ) 

            if  cf_health_condition[i, t + 1] > global_max:
                cf_health_condition[i, t + 1] = global_max
            if  cf_health_condition[i, t + 1] < global_min:
                cf_health_condition[i, t + 1] = global_min  




    outputs = {
        "adj":adj,
        "feature":feature,
        "health_condition": health_condition,
        "action_dosage": action_dosage,
        "avg_neighbor_action_dosage": avg_neighbor_action_dosage,
        "action_application_point": action_application_point,
        "avg_neighbor_action_application":avg_neighbor_action_application,
        "treatment_probs": treatment_probs,
        "noise_terms":noise_terms,

        "effect_f2y":effect_f2y,
        "avg_neighbor_effect_f2y":avg_neighbor_effect_f2y,


        "cf_health_condition": cf_health_condition,
        "cf_action_application_point":cf_action_application_point,
        "cf_avg_neighbor_action_application":cf_avg_neighbor_action_application,
        "cf_noise_terms":cf_noise_terms,

        "mean":None,
        "std":None
        }


    if norm:
        outputs = get_scaling_params(outputs)
        print ("Normalized...")
    print ("simulated shape:{}".format(health_condition.shape))

    return outputs





def get_scaling_params(sim):
    real_idx = ["health_condition", "action_dosage", "avg_neighbor_action_dosage"]

    means = {}
    stds = {}
    for k in real_idx:
        means[k] = np.mean(sim[k])
        stds[k] = np.std(sim[k])
    
    sim["mean"] = means
    sim["std"] = stds

    return sim



