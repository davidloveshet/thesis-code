


% Clopper Pearson UCB ����

function upper_bound = ClopperPearsonUCB(reward, pulls, alpha_confidence)
    binofit(reward, pulls, alpha_confidence);
end
