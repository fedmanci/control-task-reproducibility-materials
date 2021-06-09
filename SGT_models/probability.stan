# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Model "Probability"
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
  real    cp[10]; # parameters to obtain probability of success from center (see SMs)
  real    dp[19]; # parameters to obtain probability of success from positions other than center (see SMs)
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
    vector<lower = 0, upper = 1>[2]                   prob_par[n_subjects]; # probability sensitivity
    vector<lower = 0, upper = 1>[2]                   ach_par[n_subjects]; # sensitivity to previous veh history
    vector<lower = 0, upper = 15>[2]                  frequency[n_subjects]; # subjective prospective frequencty

    vector<lower = 0, upper = 1>[4]                   hyper_m; # hyper means for above parameters
    vector<lower = 0, upper = 5>[4]                   hyper_v; # hyper vars for above parameters
}

transformed parameters{

    vector[2]       pr[n_trials];

    for (n in 1:n_trials){
        pr[n,1] = 1/(1 + exp(dp[1]*frequency[sub[n],block_type[n]] + dp[2]*g1[n] + dp[3]*d11[n] + dp[4]*frequency[sub[n],block_type[n]]*d11[n] + dp[5]*frequency[sub[n],block_type[n]]*g1[n]+ dp[6]*d11[n]*g1[n] + dp[7]*(frequency[sub[n],block_type[n]]^2) + dp[8]*(g1[n]^2) + dp[9]*(d11[n]^2) + dp[10]*(frequency[sub[n],block_type[n]]^2)*d11[n] + dp[11]*(frequency[sub[n],block_type[n]]^2)*g1[n] + dp[12]*(g1[n]^2)*d11[n] + dp[13]*(g1[n]^2)*frequency[sub[n],block_type[n]] + dp[14]*(d11[n]^2)*frequency[sub[n],block_type[n]] + dp[15]*(d11[n]^2)*g1[n] + dp[16]*(frequency[sub[n],block_type[n]]^3) + dp[17]*(g1[n]^3) + dp[18]*(d11[n]^3) + dp[19]));
        pr[n,2] = 1/(1 + exp(dp[1]*frequency[sub[n],block_type[n]] + dp[2]*g2[n] + dp[3]*d21[n] + dp[4]*frequency[sub[n],block_type[n]]*d21[n] + dp[5]*frequency[sub[n],block_type[n]]*g2[n]+ dp[6]*d21[n]*g2[n] + dp[7]*(frequency[sub[n],block_type[n]]^2) + dp[8]*(g2[n]^2) + dp[9]*(d21[n]^2) + dp[10]*(frequency[sub[n],block_type[n]]^2)*d21[n] + dp[11]*(frequency[sub[n],block_type[n]]^2)*g2[n] + dp[12]*(g2[n]^2)*d21[n] + dp[13]*(g2[n]^2)*frequency[sub[n],block_type[n]] + dp[14]*(d21[n]^2)*frequency[sub[n],block_type[n]] + dp[15]*(d21[n]^2)*g2[n] + dp[16]*(frequency[sub[n],block_type[n]]^3) + dp[17]*(g2[n]^3) + dp[18]*(d21[n]^3) + dp[19]));
    }
}

model{

    vector[2]       add_opts;

    for (s in 1:n_subjects){

        prob_par[s,1] ~     normal(hyper_m[1],hyper_v[1]);
        prob_par[s,2] ~     normal(hyper_m[2],hyper_v[2]);
        ach_par[s,1] ~      normal(hyper_m[3],hyper_v[3]);
        ach_par[s,2] ~      normal(hyper_m[4],hyper_v[4]);

        # priors for frequency are hardcoded - as they are much stricter than for other params
        frequency[s,1] ~    normal(8,0.1);
        frequency[s,2] ~    normal(4,0.1);
    }

    for (n in 1:n_trials){
        if (reinforcement[n] > -40){

            add_opts[1] = prob_par[sub[n], block_type[n]] * 100 * pr[n,1] + ach_par[sub[n],block_type[n]] * 100 * ach1[n];
            add_opts[2] = prob_par[sub[n], block_type[n]] * 100 * pr[n,2] + ach_par[sub[n],block_type[n]] * 100 * ach2[n];

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

             add_opts[1] = prob_par[sub[n], block_type[n]] * 100 * pr[n,1] + ach_par[sub[n],block_type[n]] * 100 * ach1[n];
             add_opts[2] = prob_par[sub[n], block_type[n]] * 100 * pr[n,2] + ach_par[sub[n],block_type[n]] * 100 * ach2[n];

             log_lik[n] = categorical_logit_lpmf(y[n]|add_opts);
             gen_data[n] = categorical_rng(softmax(add_opts));
        }
        else{
             log_lik[n] = log_lik[n-1];
        }
    }
}
