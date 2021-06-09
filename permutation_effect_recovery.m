

% This script
% generates the p-values that are graphically presented in inset B of 
% figure 7 in  the paper.
%
% This script is intended to compute the significance of each parameter
% towards the general effect found (i.e. that I-scores correlate with the
% frequency of pursue of g+ goals).
%
% It does so by generating synthetic datasets under different random
% permutations across subjects of each single parameter, recomputing the
% correlation found as if the dataset were real, and thus obtaining an
% empirical (synthetic) distribution of datasets generated under the null
% hypothesis that the permuted parameter is irrelevant towards
% the recovery of the effect.
%
% Indeed the parameter is only relevant if only 5% or less of the synthetic
% effects found in the fictitious datasets are bigger in magnitude than the one 
% found in the real dataset (this means p = 0.05). 
%
% The values portraid in the inset are the 2-sided p-values obtained here.
%
% Each single parameter can be tested using this permutation test. This can
% be done by subsituting sub(n) (i.e. the original subject labelling) with
% perm_sub(n) (i.e. the permuted subject labelling). For instance - to test
% whether 'initialisation' matters towards the main effect, one should rewrite ONLY
% initialisation(sub(n), block_type(n)) as initialisation(perm_sub(n), block_type(n))
% leaving all other parameters intact.
%
% the final results can be found in the 'two_side_Ps' variable, expressed
% as percentages.
%
% Right now, the script is set to shuffle the distance parameter
% ('dist_sensty').

tic;

%% Imports ------------- - - - - - - - - - - -  -  -  -  -
%  --------------------- - - - - - - - - - - -  -  -  -  -

% Questionnaires.
qMatrix = importdata('./task_data/questionnaire_scores.mat');

% Task data.
data = importdata("./task_data/trialdata.mat");
sub = data(:,1);

% Probabilities for each vehicle-goal pair for each trial.
pr = load('./task_data/all_opts_probs.mat');
pr = pr.all_opts_probs;

%% Refinement of imported data, and auxiliary variables.

pr(data(:,37) > 3, :) = [];
sub(data(:,37) > 3) = [];
data(data(:,37) > 3, :) = [];

num_subjects = 35;
rewards = [data(:,6:7), data(:,6:7)];
num_trials = size(data,1);
num_permutations = 1000;

% helper functions
int_step = @(x) logical(x > 0);
tinyline = @(x,h) plot([x,x],[0,h],'r-');

first_trial = mod(data(:,2),20) == 1;
veryfirst_trial = data(:,2) == 1;
block_type = (data(:,14) == 250) + 1;

prev_vehicle = zeros(1,num_trials);
for n = 1:num_trials
    if (first_trial(n) == 0)
        prev_vehicle(n) = data(n-1,16);
    end
end

%% Import the posterior means.

posterior = readtable('posterior__vehDepRW.csv');

% posterior means
dist_sensty = [posterior.hi_dist_sensty, posterior.lo_dist_sensty];
guid_sensty = [posterior.hi_guid_sensty, posterior.lo_guid_sensty];
rew_sensty = [posterior.hi_rew_sensty, posterior.lo_rew_sensty];
lrn_gain = [posterior.hi_lrn_gain, posterior.lo_lrn_gain];
learn_rates = zeros(num_subjects,2,2);
learn_rates(:,1,:) = [posterior.hi_win_lr, posterior.hi_los_lr];
learn_rates(:,2,:) = [nan(35,1), posterior.lo_los_lr];
initialisation = [posterior.hi_initialisation, posterior.lo_initialisation];

% learned vehicle objective controllabilities
p1 = csvread(strcat('./task_data/p1.csv'),1,1);
p2 = csvread(strcat('./task_data/p2.csv'),1,1);

d11 = (data(:,8)/2 + 2) * sqrt(2);
d12 = (data(:,9)/2 + 2) * sqrt(2);
d21 = (data(:,10)/2 + 2) * sqrt(2);
d22 = (data(:,11)/2 + 2) * sqrt(2);

reinforcement = data(:,15);
high_influence = logical(data(:,14) == 125);
rew = data(:,[6,7]);

% binarised trial outcomes: +1 for a win, -1 for a loss
trial_outcome =  (data(:,15) > 0)*1 - (data(:,15) < 0)*1;

ach_g2 = zeros(num_trials, 2, 2);

for k = 1:num_permutations
    
    if k == 1
        % the first of gen_sets is the actual dataset
        perm_sub = sub; 
        permutation = 1:num_subjects;
    else
        % ...the others are permuted
        [perm_sub, permutation] = permute_subs(data(:,1), num_subjects);
    end

    gen_choice_probs = zeros(num_trials, 4);  
    
    for n = 1:num_trials
                
        if (prev_vehicle(n) == 0 && first_trial(n) == 0)
            ach_g2(n) = ach_g2(n-1);
        else
            if(veryfirst_trial(n) == 1)
                ach_g2(n,1,1) = initialisation(sub(n), 1);
                ach_g2(n,2,1) = initialisation(sub(n), 1);
                ach_g2(n,1,2) = initialisation(sub(n), 2);
                ach_g2(n,2,2) = initialisation(sub(n), 2);
            elseif(first_trial(n) == 1)
                ach_g2(n,1,1) = ach_g2(n-1,1,1)/2 + ach_g2(n-1,2,1)/2; 
                ach_g2(n,2,1) = ach_g2(n-1,1,1)/2 + ach_g2(n-1,2,1)/2; 
                ach_g2(n,1,2) = ach_g2(n-1,1,2)/2 + ach_g2(n-1,2,2)/2; 
                ach_g2(n,2,2) = ach_g2(n-1,1,2)/2 + ach_g2(n-1,2,2)/2;
            else
                if (block_type(n) == 1)
                    if (prev_vehicle(n) == 1)
                        ach_g2(n,1,1) = ach_g2(n-1,1,1) + int_step(reinforcement(n-1) - 45) * learn_rates(sub(n), block_type(n), 1) * (trial_outcome(n-1) - ach_g2(n-1,1,1)) + int_step(-trial_outcome(n-1)) * learn_rates(sub(n), block_type(n), 2) * (trial_outcome(n-1) - ach_g2(n-1,1,1)) ;
                        ach_g2(n,2,1) = ach_g2(n-1,2,1);
                    end
                    if (prev_vehicle(n) == 2)
                        ach_g2(n,2,1) = ach_g2(n-1,2,1) + int_step(reinforcement(n-1) - 45) * learn_rates(sub(n), block_type(n), 1) * (trial_outcome(n-1) - ach_g2(n-1,2,1)) + int_step(-trial_outcome(n-1)) * learn_rates(sub(n), block_type(n), 2) * (trial_outcome(n-1) - ach_g2(n-1,2,1)) ;
                        ach_g2(n,1,1) = ach_g2(n-1,1,1);
                    end
                    ach_g2(n,1,2) = ach_g2(n-1,1,2);
                    ach_g2(n,2,2) = ach_g2(n-1,2,2);
                end
                if (block_type(n) == 2)
                    if (prev_vehicle(n) == 1)
                        ach_g2(n,1,2) = ach_g2(n-1,1,2) + int_step(-trial_outcome(n-1)) * learn_rates(sub(n), block_type(n), 2) * (trial_outcome(n-1) - ach_g2(n-1,1,2)) ;
                        ach_g2(n,2,2) = ach_g2(n-1,2,2);
                    end
                    if (prev_vehicle(n) == 2)
                        ach_g2(n,2,2) = ach_g2(n-1,2,2) + int_step(-trial_outcome(n-1)) * learn_rates(sub(n), block_type(n), 2) * (trial_outcome(n-1) - ach_g2(n-1,2,2)) ;
                        ach_g2(n,1,2) = ach_g2(n-1,1,2);
                    end
                    ach_g2(n,1,1) = ach_g2(n-1,1,1);
                    ach_g2(n,2,1) = ach_g2(n-1,2,1);
                end
            end
        end
        
        options(1) = rew_sensty(sub(n),block_type(n)) * rew(n,1) + guid_sensty(sub(n), block_type(n)) * 100 * p1(n) - dist_sensty(perm_sub(n),block_type(n)) * d11(n);
        options(2) = rew_sensty(sub(n),block_type(n)) * rew(n,2) + guid_sensty(sub(n), block_type(n)) * 100 * p1(n) - dist_sensty(perm_sub(n),block_type(n)) * d12(n) + int_step(rew(n,2) - rew(n,1)) * int_step(d22(n) - d12(n)) * 10 *lrn_gain(sub(n), block_type(n)) * (ach_g2(n,1,block_type(n))); 
        options(3) = rew_sensty(sub(n),block_type(n)) * rew(n,1) + guid_sensty(sub(n), block_type(n)) * 100 * p2(n) - dist_sensty(perm_sub(n),block_type(n)) * d21(n);
        options(4) = rew_sensty(sub(n),block_type(n)) * rew(n,2) + guid_sensty(sub(n), block_type(n)) * 100 * p2(n) - dist_sensty(perm_sub(n),block_type(n)) * d22(n) + int_step(rew(n,2) - rew(n,1)) * int_step(d12(n) - d22(n)) * 10 *lrn_gain(sub(n), block_type(n)) * (ach_g2(n,2,block_type(n))); 
        
        if (reinforcement(n) > -40) % if trial was not skipped (i.e. choice time over time allowed)
            gen_choice_probs(n,:) = exp(options)/sum(exp(options));
        end
        
    end
    gen_sets{k} = gen_choice_probs;
end
    
% vectors containing empirical null distribution of effect sizes under
% permutation, in high and low influence conditions.
r_hi = zeros(1, num_permutations);
r_lo = zeros(1, num_permutations);

for k = 1:num_permutations

    for s = 1:num_subjects    
        
        % no catch or OG trials
        no_catch_or_og = logical(data(:,7) > 35);
        
        gPlusFrequency(k,s,1) = sum(sum(gen_sets{k}(sub == s & block_type == 1 & no_catch_or_og,[2,4])))/sum(sub == s & block_type == 1 & no_catch_or_og);
        gPlusFrequency(k,s,2) = sum(sum(gen_sets{k}(sub == s & block_type == 2 & no_catch_or_og,[2,4])))/sum(sub == s & block_type == 2 & no_catch_or_og);        
    end
    
    % correlation between g+ choice frequency and I-scores in high
    % influence conditions.
    A = corrcoef(gPlusFrequency(k,:,1), qMatrix(:,1));
    r_hi(k) = A(1,2);
    % correlation between g+ choice frequency and I-scores in low
    % influence conditions.
    B = corrcoef(gPlusFrequency(k,:,2), qMatrix(:,1));
    r_lo(k) = B(1,2);
    
    %% choice probabilities and choice expected rewards
    choice_probabilities = NaN(size(gen_sets{k}));
    choice_rewards = NaN(size(gen_sets{k},1),1);
    for i = 1:length(data)
        if reinforcement(i) > -40
            % model-generated: average prob success on i-th trial
            choice_probabilities(i) = pr(i,:) * gen_sets{k}(i,:)';
            % model-generated: average reward on i-th trial
            choice_rewards(i) = (pr(i,:).*rewards(i,:) + (1 - pr(i,:)).*(-15)) * gen_sets{k}(i,:)';
        end
    end

    for s = 1:num_subjects
        synthetic_choice_probs(k,s,1) = nanmean(choice_probabilities(sub == s & high_influence));
        synthetic_choice_probs(k,s,2) = nanmean(choice_probabilities(sub == s & not(high_influence)));    
        synthetic_money_earned(k,s,1) = nansum(choice_rewards(high_influence & sub == s));
        synthetic_money_earned(k,s,2) = nansum(choice_rewards(not(high_influence) & sub == s));
    end
    
    B = corrcoef(synthetic_money_earned(k,:,2)', qMatrix(:,3));
    r2(k) = B(1,2);
end
    
figure,
histogram(r_hi), hold on, tinyline(r_hi(1),50),
title('original labelling (red line) vs. null distribution [High Influence]')
figure,
histogram(r_lo), hold on, tinyline(r_lo(1),50),
title('original labelling (red line) vs. null distribution [Low Influence]')
two_side_Ps(1) = 100*sum(abs(r_hi(2:end)) > abs(r_hi(1)))./num_permutations;
two_side_Ps(2) = 100*sum(abs(r_lo(2:end)) > abs(r_lo(1)))./num_permutations;

% rel = zeros(1,num_subjects);
% for s = 1:num_subjects
%     C = corrcoef(lr_change(:,s), synthetic_money_earned(:,s,2) - synthetic_money_earned(1,s,2));
%     rel(s) = C(1,2);
% end
% 
% igc = data(:,17);
% vgc = data(:,6).* (igc == 1) + data(:,7).* (igc == 2);
% 
% 
% data_gplus = zeros(num_subjects, 2);
% data_choiceprobs = zeros(num_subjects, 2);
% data_allmoney = zeros(num_subjects, 2);
% modgen_gplus = zeros(num_subjects, 2);
% 
% ivc = data(:,16);
% wcv = (data(:,8) == 36 & data(:,9) == 36)*1 + ...
%     (data(:,10) == 36 & data(:,11) == 36)*2; % 1:v1 is central, 2:v2 is central, 3:both.
% ccv = (ivc == 1 & wcv == 1) | (ivc == 2 & wcv == 2) | wcv == 3; % 1 if chose central v.
% vvc = data(:,4).* (ivc == 1) + data(:,5).* (ivc == 2);
% 
% for s = 1:num_subjects     
%     
%     % no catch or og trials
%     no_catch_or_og = not(data(:,7) == 25 | data(:,7) == 35);
%     
%     data_rightmoveperc(s,1) = nanmean(data(sub == s & high_influence, 33)./data(sub == s & high_influence, 32));
%     data_rightmoveperc(s,2) = nanmean(data(sub == s & not(high_influence), 33)./data(sub == s & not(high_influence), 32));
%     
%     data_choiceprobs(s,1) = nanmean(emp_choice_prob(sub == s & high_influence & data(:,6) == 65));
%     data_choiceprobs(s,2) = nanmean(emp_choice_prob(sub == s & not(high_influence) & data(:,6) == 65));
%     
%     data_gplus(s,1) = sum(vgc(sub == s & high_influence & no_catch_or_og) > 45)/sum(sub == s & high_influence & no_catch_or_og);
%     data_gplus(s,2) = sum(vgc(sub == s & not(high_influence) & no_catch_or_og) > 45)/sum(sub == s & not(high_influence) & no_catch_or_og);
%     data_gplus_equalguid(s,1) = sum(vgc(sub == s & high_influence & no_catch_or_og & lg_v(2,:)') > 45)/sum(sub == s & high_influence & no_catch_or_og & lg_v(2,:)');
%     data_gplus_equalguid(s,2) = sum(vgc(sub == s & not(high_influence) & no_catch_or_og & lg_v(2,:)') > 45)/sum(sub == s & not(high_influence) & no_catch_or_og & lg_v(2,:)');
%        
%     data_vplus(s,1,1) = sum(ivc(sub == s & high_influence & ~lg_v(3,:)' & no_catch_or_og) == 2)/sum(sub == s & high_influence & ~lg_v(3,:)' & no_catch_or_og);
%     data_vplus(s,2,1) = sum(ivc(sub == s & not(high_influence) & ~lg_v(3,:)' &no_catch_or_og) == 2)/sum(sub == s & not(high_influence) & ~lg_v(3,:)' & no_catch_or_og);
%     data_vplus(s,1,2) = sum(ivc(sub == s & high_influence & lg_v(3,:)' &no_catch_or_og) == 2)/sum(sub == s & high_influence & lg_v(3,:)' & no_catch_or_og);
%     data_vplus(s,2,2) = sum(ivc(sub == s & not(high_influence) & lg_v(3,:)' &no_catch_or_og) == 2)/sum(sub == s & not(high_influence) & lg_v(3,:)' & no_catch_or_og);
% 
%  
%     % model generated
%     modgen_gplus(s,1) = sum(sum(gen_sets{1}(sub == s & block_type == 1 & no_catch_or_og,[2,4])))/sum(sub == s & block_type == 1 & no_catch_or_og);
%     modgen_gplus(s,2) = sum(sum(gen_sets{1}(sub == s & block_type == 2 & no_catch_or_og,[2,4])))/sum(sub == s & block_type == 2 & no_catch_or_og);
% 
%     modgen_gplus_equalguid(s,1) = sum(sum(gen_sets{1}(sub == s & block_type == 1 & no_catch_or_og & lg_v(2,:)',[2,4])))/sum(sub == s & block_type == 1 & no_catch_or_og & lg_v(2,:)');
%     modgen_gplus_equalguid(s,2) = sum(sum(gen_sets{1}(sub == s & block_type == 2 & no_catch_or_og & lg_v(2,:)',[2,4])))/sum(sub == s & block_type == 2 & no_catch_or_og & lg_v(2,:)');
% 
%     modgen_vplus(s,1) = sum(sum(gen_sets{1}(sub == s & block_type == 1 & lg_v(3,:)' ,[3,4])))/sum(sub == s & block_type == 1 & lg_v(3,:)' );
%     modgen_vplus(s,2) = sum(sum(gen_sets{1}(sub == s & block_type == 2 & lg_v(3,:)' ,[3,4])))/sum(sub == s & block_type == 2 & lg_v(3,:)' );
% end
% 
% toc;
% % clearvars -except qMatrix r theP*
% 
% subplot(1,2,1),
%     plot(modgen_gplus_equalguid(:,1), data_gplus_equalguid(:,1),'ko'), hold on,
%     plot(modgen_gplus_equalguid(:,2), data_gplus_equalguid(:,2),'bo'),
%     plot(0:0.1:0.5, 0:0.1:0.5, 'r-')
% subplot(1,2,2),
%     plot(modgen_vplus(:,1), data_vplus(:,1),'ko'), hold on,
%     plot(modgen_vplus(:,2), data_vplus(:,2),'bo')  
%     plot(0.4:0.1:0.8, 0.4:0.1:0.8, 'r-')    


