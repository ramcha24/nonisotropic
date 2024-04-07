import foolbox.attacks as fa

epsilons = {'fixed_l2': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            'fixed_linf': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'minimum_size': None}

fixed_size_attacks_l2 = {
    'l2_pgd': fa.L2ProjectedGradientDescentAttack(steps=100),
    'l2_bia': fa.L2BasicIterativeAttack(steps=100),
    'l2_fgm': fa.L2FastGradientAttack()
}

fixed_size_attacks_linf = {
    'linf_pgd': fa.LinfProjectedGradientDescentAttack(steps=100),
    'linf_bia': fa.LinfBasicIterativeAttack(steps=100),
    'linf_fgm': fa.LinfFastGradientAttack()
}

minimization_attacks = {
    'min_ddn': fa.DDNAttack(init_epsilon=3.0, steps=500, gamma=0.002),
    'min_cw': fa.L2CarliniWagnerAttack(),
    'min_deepfool': fa.L2DeepFoolAttack(),
}


def get_possible_attacks():
    pos = ''
    for attack_str in fixed_size_attacks_l2.keys():
        pos = pos + '\n' + attack_str
    for attack_str in fixed_size_attacks_linf.keys():
        pos = pos + '\n' + attack_str
    for attack_str in minimization_attacks.keys():
        pos = pos + '\n' + attack_str

    return pos