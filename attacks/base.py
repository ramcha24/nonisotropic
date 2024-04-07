import torch
import eagerpy as ep
from foolbox import PyTorchModel
import numpy as np

from platforms.platform import get_platform

def extract_conv_patches(batch_inputs, kernel_shape):
    # returns shape : (b, c , kernel, kernel, h_new, w_new) where (c, kernel, kernel) is the dimension of an individual patch of which there are h_new*w_new in number.
    batch_size, c_old, h_old, w_old = batch_inputs.shape
    h_new = h_old - kernel_shape + 1
    w_new = w_old - kernel_shape + 1

    patches = torch.zeros(batch_size, c_old, kernel_shape, kernel_shape, h_new, w_new).cuda()

    for i in range(0, h_new):
        for j in range(0, w_new):
            patches[:, :, :, :, i, j] = batch_inputs[:, :, i:i + kernel_shape, j:j + kernel_shape]

    return patches


def distance(attack_norm):
    if attack_norm == '2':
        return ep.norms.l2
    elif attack_norm == '1':
        return ep.norms.l1
    elif attack_norm == 'inf':
        return ep.norms.linf
    else:
        raise ValueError("No recognized distance wrt norm - {}".format(attack_norm))


class FixedSizeAttack:
    def __init__(self, attack_fn, epsilons, attack_norm, num_trials):
        self.attack_fn = attack_fn
        self.epsilons = epsilons
        self.name = str(attack_fn).partition('(')[0]
        self.distance_fn = distance(attack_norm)
        self.num_trials = num_trials

    def generate_adversarial_example(self,
                                     model,
                                     examples,
                                     labels,
                                     margins,
                                     conv_flag: bool = False):
        kernel_shape = None
        if conv_flag:
            kernel_shape = 3  # hardcoded for now.

        lower = torch.min(examples).item()
        upper = torch.max(examples).item()
        foolbox_model = PyTorchModel(model, bounds=(lower, upper), preprocessing=None)

        examples_ep, labels_ep = ep.astensors(examples, labels)

        # assuming that examples and labels are eager py tensors
        batch_size = len(examples)
        best_adv = examples.clone().detach()
        # best_adv = torch.zeros_like(examples.raw).to(device=get_platform().torch_device)  # dimensions : batch_size x image_dims
        best_adv_radius = [np.inf] * batch_size  # dimensions : batch_size x 1
        found_adv = [False] * batch_size

        assert batch_size == len(margins)

        for j in range(0, batch_size):
            if margins[j] < 0:
                best_adv_radius[j] = 0.0
                found_adv[j] = True
                # best_adv[j] = examples[j].raw
        #self.num_trials
        for i in range(1, 2):
            print('\n Trial number {}'.format(i))
            raw, clip, _ = self.attack_fn(foolbox_model, examples_ep, labels_ep, epsilons=self.epsilons)
            for k in range(0, len(self.epsilons)):
                print('Epsilon of attack {}'.format(self.epsilons[k]))
                adv_output = model(clip[k].raw)
                adv_class = torch.eq(labels, adv_output.argmax(dim=1))

                if conv_flag:
                    print("Inside diff patch computation")
                    diff = (clip[k] - examples_ep).raw
                    diff_patches = extract_conv_patches(diff, kernel_shape)
                    distances = torch.zeros(diff.shape[0])

                    for index in range(0, diff_patches.shape[0]):
                        for u in range(0, diff_patches.shape[2]):
                            for v in range(0, diff_patches.shape[3]):
                                temp = torch.linalg.vector_norm(diff_patches[index, :, :, :, u, v], ord=2).item()
                                if temp > distances[index]:
                                    distances[index] = temp
                else:
                    distances = self.distance_fn((clip[k] - examples_ep).reshape((batch_size, 784)), axis=1)

                for j in range(0, batch_size):
                    print('\n Original margin {}, Current found_adv {}, Adv Classification result {}, adversarial corruption size {}, current best_adv_radius {}'
                          .format(margins[j], found_adv[j], adv_class[j], distances[j].item(), best_adv_radius[j]))
                    if margins[j] >= 0:
                        if (adv_class[j] == False) and best_adv_radius[j] > distances[j].item():
                            # print('Found new improved adversary for example {}'.format(j))
                            best_adv_radius[j] = distances[j].item()
                            best_adv[j] = clip[k][j].raw
                            found_adv[j] = True

        # some logic to find the new improved robust accuracy along with mean adversarial radius

        best_adv = best_adv.to(device=get_platform().torch_device)
        best_adv_radius = torch.Tensor(best_adv_radius).to(device=get_platform().torch_device)

        return best_adv, best_adv_radius, found_adv

    def get_name(self): return self.name


class MinimizationAttack:
    def __init__(self, attack_fn, attack_norm, num_trials):
        self.attack_fn = attack_fn
        self.name = str(attack_fn).partition('(')[0]
        self.distance_fn = distance(attack_norm)
        self.num_trials = num_trials

    def generate_adversarial_example(self,
                                     model,
                                     examples,
                                     labels,
                                     margins,
                                     conv_flag: bool = False):
        print('Running Attack {}'.format(self.get_name()))
        kernel_shape = None
        if conv_flag:
            kernel_shape = 3 # hardcoded for now.

        lower = torch.min(examples).item()
        upper = torch.max(examples).item()
        foolbox_model = PyTorchModel(model, bounds=(lower, upper), preprocessing=None)

        examples_ep, labels_ep = ep.astensors(examples, labels)

        batch_size = len(examples_ep)
        best_adv = examples.clone().detach()
        # best_adv = torch.zeros_like(examples.raw).to(device=get_platform().torch_device)  # dimensions : batch_size x image_dims
        best_adv_radius = [np.inf] * batch_size  # dimensions : batch_size x 1
        found_adv = [False] * batch_size

        assert batch_size == len(margins)

        for j in range(0, batch_size):
            if margins[j] < 0:
                best_adv_radius[j] = 0.0
                found_adv[j] = True
                # best_adv[j] = examples[j].raw

        for i in range(1, 2):
            print('\n Trial number {}'.format(i))
            adv_raw, adv_clip, _ = self.attack_fn(foolbox_model, examples_ep, labels_ep, epsilons=None)
            adv_output = model(adv_raw.raw)
            adv_class = torch.eq(labels, adv_output.argmax(dim=1))

            if conv_flag:
                print("Inside diff patch computation")
                diff = (adv_raw - examples_ep).raw
                diff_patches = extract_conv_patches(diff, kernel_shape)
                distances = torch.zeros(diff.shape[0])

                for index in range(0, diff_patches.shape[0]):
                    for u in range(0, diff_patches.shape[2]):
                        for v in range(0, diff_patches.shape[3]):
                            temp = torch.linalg.vector_norm(diff_patches[index, :, :, :, u, v], ord=2).item()
                            if temp > distances[index]:
                                distances[index] = temp
            else:
                distances = self.distance_fn((adv_raw - examples_ep).reshape((batch_size, 784)), axis=1)

            for j in range(0, batch_size):
                print(
                    '\n Original margin {}, Current found_adv {}, Adv Classification result {}, adversarial corruption size {}, current best_adv_radius {}'
                    .format(margins[j], found_adv[j], adv_class[j], distances[j].item(), best_adv_radius[j]))
                if margins[j] >= 0:
                    if (adv_class[j] == False) and best_adv_radius[j] > distances[j].item():
                        #print('Found new improved adversary for example {}'.format(j))
                        best_adv_radius[j] = distances[j].item()
                        best_adv[j] = adv_raw[j].raw
                        found_adv[j] = True

        best_adv = best_adv.to(device=get_platform().torch_device)
        best_adv_radius = torch.Tensor(best_adv_radius).to(device=get_platform().torch_device)

        return best_adv, best_adv_radius, found_adv

    def get_name(self): return self.name
