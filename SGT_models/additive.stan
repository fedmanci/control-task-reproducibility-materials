# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Model "Additive"
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#
# This model can be ran using any stan instantiation (this has been run in both
# pyStan and Rstan without problems).
#
# We suggest running the model with 1000 warmup samples, 2400 iterations, a tree-depth of 10,
# and an adapt-delta of 0.999999.
#
#
# coded by Federico Mancinelli
#

data {
  // real    cp[10]; # parameters to obtain probability of success from center (see SMs)
  // real    dp[19]; # parameters to obtain probability of success from positions other than center (see SMs)
  int     n_trials;
  int     n_subjects;
  real    d11[n_trials]; # distance between vehicle 1 and goal
  real    d21[n_trials]; # distance between vehicle 2 and goal
  real    g1[n_trials]; # guidability, vehicle 1
  real    g2[n_trials]; # guidability, vehicle 2
  real    ach1[n_trials]; # H_{16} for vehicle 1
  real    ach2[n_trials]; # H_{16} for vehicle 2
  int     sub[n_trials]; # subject ID
  int     block_type[n_trials]; # influence condition
  int     reinforcement[n_trials]; # money gain or loss at trial n
  int     y[n_trials]; # outcome (1: vehicle 1; 2: vehicle 2)
}

parameters{
    vector<lower = 0, upper = 1>[2]                 dist_par[n_subjects]; # distance sensitivity
    vector<lower = 0, upper = 1>[2]                 veh_par[n_subjects]; # guidability sensitivity
    vector<lower = 0, upper = 1>[2]                 ach_par[n_subjects]; # achievability sensitivity

    vector<lower = 0, upper = 1>[6]                 hyper_m; # hyper means for above parameters
    vector<lower = 0, upper = 5>[6]                 hyper_v; # hyper vars for above parameters
}

model{

    vector[2]       add_opts;

    for (s in 1:n_subjects){

        dist_par[s,1] ~     normal(hyper_m[1],hyper_v[1]);
        dist_par[s,2] ~     normal(hyper_m[2],hyper_v[2]);
        veh_par[s,1] ~      normal(hyper_m[3],hyper_v[3]);
        veh_par[s,2] ~      normal(hyper_m[4],hyper_v[4]);
        ach_par[s,1] ~    normal(hyper_m[5],hyper_v[5]);
        ach_par[s,2] ~    normal(hyper_m[6],hyper_v[6]);
    }

    for (n in 1:n_trials){

        if (reinforcement[n] > -40){

            add_opts[1] = -d11[n] * dist_par[sub[n],block_type[n]] + ach_par[sub[n],block_type[n]]*100*ach1[n] + g1[n]*100*veh_par[sub[n],block_type[n]];
            add_opts[2] = -d21[n] * dist_par[sub[n],block_type[n]] + ach_par[sub[n],block_type[n]]*100*ach2[n] + g2[n]*100*veh_par[sub[n],block_type[n]];

            target += categorical_logit_lpmf(y[n]|add_opts);
        }
    }
}

generated quantities{

    vector[n_trials]  log_lik;
    vector[n_trials]  gen_data;
    vector[2]         add_opts;

    for (n in 1:n_trials){
        if (reinforcement[n] > -40){

            add_opts[1] = -d11[n] * dist_par[sub[n],block_type[n]] + ach_par[sub[n],block_type[n]]*100*ach1[n] + g1[n]*100*veh_par[sub[n],block_type[n]];
            add_opts[2] = -d21[n] * dist_par[sub[n],block_type[n]] + ach_par[sub[n],block_type[n]]*100*ach2[n] + g2[n]*100*veh_par[sub[n],block_type[n]];

            log_lik[n] = categorical_logit_lpmf(y[n]|add_opts);
            gen_data[n] = categorical_rng(softmax(add_opts));
        }
        else{
            log_lik[n] = log_lik[n-1];
            gen_data[n] = gen_data[n-1];
        }
    }
}
