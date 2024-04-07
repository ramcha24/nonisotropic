from lipschitz.SLL_feedforward import SLL_feedforward_eval
from lipschitz.SLL_conv import SLL_conv_eval

registered_estimators = {'SLL_ffd': SLL_feedforward_eval,
                         'SLL_conv': SLL_conv_eval}
