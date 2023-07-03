


% Clopper Pearson UCB ����

function upperbound = ClopperPearsonUCB(reward, pulls, alpha_confidence)
    [PHAT, PCI] = binofit(reward, pulls, alpha_confidence);
    upperbound = PCI(2);
end
