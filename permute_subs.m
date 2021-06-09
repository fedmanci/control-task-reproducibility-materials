function [permuted_subs, permutation] = permute_subs(subs, num_subjects)
%PERMUTE_SUBS Summary of this function goes here
%   Detailed explanation goes here

    permuted_subs = subs;
    permutation = randperm(num_subjects);

    for i = 1:num_subjects
        permuted_subs(subs == i) = permutation(i);
    end

end


