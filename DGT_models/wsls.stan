
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Model "Win-stay Lose-shift"
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
    vector<lower = 0, upper = 5>[2]                     rew_par[num_subjects]; # alpha_{r}
    vector<lower = 0, upper = 5>[2]                     dst_par[num_subjects]; # alpha_{d}
    vector<lower = 0, upper = 5>[2]                     veh_par[num_subjects]; # alpha_{v}
    vector<lower = 0, upper = 5>[6]                     hyper_m; # hyper means for parameters above
    vector<lower = 0.001, upper = 10>[6]                 hyper_v; # hyper vars for parameters above

    vector<lower = 0.1, upper = 10>[num_subjects]        w_par; # omega_w (stay incentive in case of win)
    vector<lower = 0.1, upper = 10>[2]                   l_par[num_subjects]; # omega_l (switch incentive in case of loss)
    vector<lower = 0.01, upper = 5>[2]                   hyper_w; # hyper shape and scale for omega_w
    vector<lower = 0.01, upper = 5>[4]                   hyper_l; # hyper shape and scale for omega_l
}

transformed parameters{

    real p1[num_trials];
    real p2[num_trials];
    real z[4,num_trials];
    real ach_g2[num_trials,2,2];

    for (n in 1:num_trials){
        if (prev_vehicle[n] == 0 && first_trial[n] == 0){
            ach_g2[n,1,1] = 0;
            ach_g2[n,2,1] = 0;
            ach_g2[n,1,2] = 0;
            ach_g2[n,2,2] = 0;
        }
        else{
            if(first_trial[n] == 1){
                ach_g2[n,1,1] = 0;
                ach_g2[n,2,1] = 0;
                ach_g2[n,1,2] = 0;
                ach_g2[n,2,2] = 0;
            }
            else{
                if (block_type[n] == 1){
                    if (prev_vehicle[n] == 1){
                        ach_g2[n,1,1] = int_step(reinforcement[n-1] - 45) * w_par[sub[n]] - int_step(-trial_outcome[n-1]) * l_par[sub[n],block_type[n]];
                        ach_g2[n,2,1] = 0;
                    }
                    if (prev_vehicle[n] == 2){
                        ach_g2[n,2,1] = int_step(reinforcement[n-1] - 45) * w_par[sub[n]] - int_step(-trial_outcome[n-1]) * l_par[sub[n],block_type[n]];
                        ach_g2[n,1,1] = 0;
                    }
                }
                if (block_type[n] == 2){
                    if (prev_vehicle[n] == 1){
                        ach_g2[n,1,2] = -int_step(-trial_outcome[n-1]) * l_par[sub[n],block_type[n]];
                        ach_g2[n,2,2] = 0;
                    }
                    if (prev_vehicle[n] == 2){
                        ach_g2[n,2,2] = -int_step(-trial_outcome[n-1]) * l_par[sub[n],block_type[n]];;
                        ach_g2[n,1,2] = 0;
                    }
                }
            }
        }

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
        rew_par[s,2] ~ normal(hyper_m[2], hyper_v[2]);
        veh_par[s,1] ~ normal(hyper_m[3], hyper_v[3]);
        veh_par[s,2] ~ normal(hyper_m[4], hyper_v[4]);
        dst_par[s,1] ~ normal(hyper_m[5], hyper_v[5]);
        dst_par[s,2] ~ normal(hyper_m[6], hyper_v[6]);

        w_par[s] ~ gamma(hyper_w[1], hyper_w[2]);
        l_par[s,1] ~ gamma(hyper_l[1], hyper_l[2]);
        l_par[s,2] ~ gamma(hyper_l[3], hyper_l[4]);
    }

    for (n in 1:num_trials){
        options[1] = rew_par[sub[n],block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d11[n];
        options[2] = rew_par[sub[n],block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d12[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d22[n] - d12[n]) * ach_g2[n,1,block_type[n]];
        options[3] = rew_par[sub[n],block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d21[n];
        options[4] = rew_par[sub[n],block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d22[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d12[n] - d22[n]) * ach_g2[n,2,block_type[n]];

        if (reinforcement[n] > -40){
            target += categorical_logit_lpmf(outcome[n]|options);
        }
    }
}

generated quantities{

    vector[num_trials] log_lik;
    vector[num_trials] gen_data;
    vector[4] options;

    for (n in 1:num_trials){
        options[1] = rew_par[sub[n],block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d11[n];
        options[2] = rew_par[sub[n],block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p1[n] - dst_par[sub[n],block_type[n]] * d12[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d22[n] - d12[n]) * ach_g2[n,1,block_type[n]];
        options[3] = rew_par[sub[n],block_type[n]] * rew[n,1] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d21[n];
        options[4] = rew_par[sub[n],block_type[n]] * rew[n,2] + veh_par[sub[n], block_type[n]] * 100 * p2[n] - dst_par[sub[n],block_type[n]] * d22[n] + int_step(rew[n,2] - rew[n,1]) * int_step(d12[n] - d22[n]) * ach_g2[n,2,block_type[n]];

        if (reinforcement[n] > -40){
            log_lik[n] = categorical_logit_lpmf(outcome[n]|options);
            gen_data[n] = categorical_rng(softmax(options));
        }
        else{
            log_lik[n] = log_lik[n-1];
        }
    }
}
