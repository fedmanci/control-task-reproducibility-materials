
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Model "Bias"
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
    int     num_trials;
    real    d11[num_trials]; # distance (Euclidean) from vehicle 1 to goal g-
    real    d12[num_trials]; # distance (Euclidean) from vehicle 1 to goal g+
    real    d21[num_trials]; # distance (Euclidean) from vehicle 2 to goal g-
    real    d22[num_trials]; # distance (Euclidean) from vehicle 2 to goal g+
    int     num_subjects;
    int     prev_vehicle[num_trials]; # previous vehicle chosen
    int     first_trial[num_trials]; # 1: first trial of every block. 0: every other trial.
    int     veryfirst_trial[num_trials]; # 1: first trial of the task. 0: every other trial.
    int     vehicle[num_trials]; # which vehicle (1, or 2)
    int     block_num[num_trials]; # block number
    int     sub[num_trials]; # subject ID
    int     block_type[num_trials]; # influence condition (1, high; 2, low)
    real    rew[num_trials,2]; # rewards indicated on goals in trial n.
    int     n_good[num_trials]; # number of good moves registered in trial n
    int     n_tot[num_trials]; # number of bad moves registered in trial n
    int     outcome[num_trials]; # subject chose 1: vehicle 1 and goal 1; 2: vehicle 1 and goal 2; 1: vehicle 2 and goal 1; 1: vehicle 2 and goal 2;
    int     trial_outcome[num_trials]; # 1: win, -1: loss.
    int     reinforcement[num_trials]; # money gained, or lost, at trial n.
}

parameters{
    vector<lower = 0, upper = 5>[2]                  rew_par[num_subjects]; # alpha_{r}
    vector<lower = 0, upper = 5>[2]                  dst_par[num_subjects]; # alpha_{d}
    vector<lower = 0, upper = 5>[2]                  veh_par[num_subjects]; # alpha_{v}
    vector<lower = 0, upper = 5>[6]                  hyper_m; # hyper means for parameters above
    vector<lower = 0.001, upper = 10>[8]             hyper_v; # hyper vars for parameters above (and bias)

    vector<lower = -5, upper = 5>[2]               bias_par[num_subjects]; # beta
    vector<lower = -5, upper = 5>[2]               hyper_bias_m; # hyper mean for bias
}

transformed parameters{

    real p1[num_trials];
    real p2[num_trials];
    real z[4,num_trials];

    for (n in 1:num_trials){

        if( first_trial[n] == 1 ){
                z[1,n] = 1;
                z[2,n] = 1;
                z[3,n] = 2;
                z[4,n] = 2;
        }
        else{
            if (vehicle[n-1] == 1 && reinforcement[n-1] > -40){
                z[1,n] = z[1,n-1] + n_good[n-1];
                z[2,n] = z[2,n-1];
                z[3,n] = z[3,n-1] + n_tot[n-1];
                z[4,n] = z[4,n-1];
            }
            else if (vehicle[n-1] == 2 && reinforcement[n-1] > -40){
                z[1,n] = z[1,n-1];
                z[2,n] = z[2,n-1] + n_good[n-1];
                z[3,n] = z[3,n-1];
                z[4,n] = z[4,n-1] + n_tot[n-1];
            }
            else {
                z[1,n] = z[1,n-2];
                z[2,n] = z[2,n-2];
                z[3,n] = z[3,n-2];
                z[4,n] = z[4,n-2];
            }
        }
        p1[n] =  1.33*(z[1,n]/z[3,n]) - 0.33;
        p2[n] =  1.33*(z[2,n]/z[4,n]) - 0.33;
    }
}

model{

    vector[4] options;

    for (s in 1:num_subjects){
        rew_par[s,1] ~ normal(hyper_m[1], hyper_v[1]);
        veh_par[s,1] ~ normal(hyper_m[2], hyper_v[2]);
        dst_par[s,1] ~ normal(hyper_m[3], hyper_v[3]);
        rew_par[s,2] ~ normal(hyper_m[4], hyper_v[4]);
        veh_par[s,2] ~ normal(hyper_m[5], hyper_v[5]);
        dst_par[s,2] ~ normal(hyper_m[6], hyper_v[6]);
        bias_par[s,1] ~ normal(hyper_bias_m[1], hyper_v[7]);
        bias_par[s,2] ~ normal(hyper_bias_m[2], hyper_v[8]);
    }

    for (n in 1:num_trials){

        options[1] = rew_par[sub[n], block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d11[n];
        options[2] = rew_par[sub[n], block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d12[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d22[n] - d12[n] + 1) * bias_par[sub[n], block_type[n]];
        options[3] = rew_par[sub[n], block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d21[n];
        options[4] = rew_par[sub[n], block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d22[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d12[n] - d22[n] + 1) * bias_par[sub[n], block_type[n]];

        if (reinforcement[n] > -40){
            target += categorical_logit_lpmf(outcome[n]|options);
        }
    }
}

generated quantities{

    vector[num_trials] log_lik;
    vector[4] options;

    for (n in 1:num_trials){

        options[1] = rew_par[sub[n], block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d11[n];
        options[2] = rew_par[sub[n], block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d12[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d22[n] - d12[n] + 1) * bias_par[sub[n], block_type[n]];
        options[3] = rew_par[sub[n], block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d21[n];
        options[4] = rew_par[sub[n], block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d22[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d12[n] - d22[n] + 1) * bias_par[sub[n], block_type[n]];

        if (reinforcement[n] > -40){
            log_lik[n] = categorical_logit_lpmf(outcome[n]|options);
        }
        else{
            log_lik[n] = log_lik[n-1];
        }
    }
}
