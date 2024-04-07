import torch.nn as nn
from collections import OrderedDict


class ForwardHookModel(nn.Module):
    def __init__(self, trained_model, output_layers, *args):
        super(ForwardHookModel, self).__init__(*args)
        self.output_layers = output_layers
        self.selected_out = OrderedDict()
        self.pretrained = trained_model
        self.fhooks = []

        self.selected_out.keys()

        for index, layer_name in enumerate(list(self.pretrained._modules.keys())):
            # print('\n index : {} \t layer name : {}'.format(index, layer_name))
            if index in self.output_layers:
                self.fhooks.append(
                    getattr(self.pretrained, layer_name).register_forward_hook(self.forward_hook(layer_name)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out