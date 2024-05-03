#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import clear_output


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# In[4]:


assert torch.cuda.is_available()
cuda = torch.device('cuda')


# In[5]:


# Handcrafted data domain specified by corners
x_start = 1.0
x_end = 5.5
y_start = -1.5
y_end = 1.5
corners = [x_start, x_end, y_start, y_end]

# Handcrafted reference points
x0 = [3.8, 0.9]
x1 = [2.7, 0.361]
x2 = [4.94, 0.452]
xtil = [1.4, -0.276]
reference_points = [x0, x1, x2, xtil]

# stale line from x0 to xtil
#x7 = np.linspace(1.4,3.8,10)
#y7 = 0.49*x7 - 0.962


# In[6]:


# Handcrafter classifiers 
def h_star_bnd(t : torch.Tensor):
    return torch.sin(t+0.5)

def h_star(t: torch.Tensor):
    assert len(t.shape) == 2 # shape is batch x inp_dimensions 
    assert t.shape[1] == 2
    out = torch.zeros_like(t)
    
    for index in range(t.shape[0]):
        out[index][0] = h_star_bnd(t[index][0])
        out[index][1] = t[index][1] 

    return out 

def h_1_bnd(t: torch.Tensor):
    return torch.sin(t+0.5)  - 0.75*torch.cos(t+0.5) - 0.2*torch.sin(3*(t+0.5)) - 0.1 

def h_1(t: torch.Tensor):
    assert len(t.shape) == 2 # shape is batch x inp_dimensions 
    assert t.shape[1] == 2
    out = torch.zeros_like(t)
    
    for index in range(t.shape[0]):
        out[index][0] = h_1_bnd(t[index][0])
        out[index][1] = t[index][1] 

    return out 


def h_2_bnd(t: torch.Tensor):
    return torch.sin(t+0.5) + 0.07*torch.sin(5*(t+0.5)) + 0.15*((t+0.5)**2) - 0.49*(t+0.5) + 0.2 - 1*(F.relu(t-4.3))**2

def h_2(t: torch.Tensor):
    assert len(t.shape) == 2 # shape is batch x inp_dimensions 
    assert t.shape[1] == 2
    out = torch.zeros_like(t)
    
    for index in range(t.shape[0]):
        out[index][0] = h_2_bnd(t[index][0])
        out[index][1] = t[index][1] 

    return out   

def predict_label(classifier, input):
    dim_flag = False 
    if input.dim() == 1:
        input = input.unsqueeze(0)
        dim_flag = True 
    out = classifier(input)
    labels = torch.argmax(out, dim=1)
    if dim_flag:
        return labels[0].item()
    else: 
        return labels 


# In[7]:


# only needed for plotting so returning numpy arrays
def generate_decision_boundary(corners):
    x_start, x_end = corners[0], corners[1]
    x = torch.linspace(x_start,x_end,100)
    y_star = h_star_bnd(x).detach().cpu()
    y_1 = h_1_bnd(x).detach().cpu()
    y_2 = h_2_bnd(x).detach().cpu()
    return x.numpy(), y_star.numpy(), y_1.numpy(), y_2.numpy()


def generate_data_domain(corners):
    x_start, x_end, y_start, y_end = corners[0], corners[1], corners[2], corners[3]
    x_left = np.ones(1000)*x_start
    y_left = np.linspace(y_start,y_end,1000)

    x_right = torch.ones(1000)*x_end
    y_right = np.linspace(y_start,y_end,1000)

    x_down = np.linspace(x_start,x_end,100)
    y_down = np.ones(100)*y_start

    x_up = np.linspace(x_start,x_end,100)
    y_up = np.ones(100)*y_end

    return x_left, y_left, x_right, y_right, x_down, y_down, x_up, y_up


# In[8]:


def plot_synthetic(corners, reference_points, show_h1=False, show_h2=False):
    x0, x1, x2, xtil = reference_points[0], reference_points[1], reference_points[2], reference_points[3]
    x_left, y_left, x_right, y_right, x_down, y_down, x_up, y_up = generate_data_domain(corners)
    x, y_star, y_1, y_2 = generate_decision_boundary(corners)

    # plot the data domain
    plt.plot(x_left, y_left, color="black", linewidth=3)
    plt.plot(x_right, y_right, color="black", linewidth=3)
    plt.plot(x_down, y_down, color="black", linewidth=3)
    plt.plot(x_up, y_up, color="black", linewidth=3)
    
    # plot true labeling decision boundary 
    plt.plot(x, y_star, label=r'$h^{\star}$', linewidth=3)
    plt.fill_between(x, y1=y_star, y2=y_down, color="red", alpha=0.05)
    plt.fill_between(x, y1=y_star, y2=y_up, color="green", alpha=0.05)

    if show_h1:
        plt.plot(x, y_1, color='magenta', label="h1", linestyle="dashed")
        plt.fill_between(x, y1=y_1, y2=y_star, color= "magenta", alpha= 0.05, hatch='///')
    
    if show_h2:
        plt.plot(x, y_2, color='purple', label="h2", linestyle="dashed")
        plt.fill_between(x, y1=y_2, y2=y_star, color= "purple", alpha= 0.05, hatch='///')
    
    # plot reference points
    plt.plot(x0[0], x0[1], color="red", marker=r"$x$", markersize=9)
    plt.plot(x1[0], x1[1], color="red", marker=r"$x_1$", markersize=12)
    plt.plot(x2[0], x2[1], color="red", marker=r"$x_2$", markersize=12)
    plt.plot(xtil[0], xtil[1], color='red', marker=r'$\tilde{x}$', markersize=12)
    # plt.plot([4.7],[0.1], color='red', marker=r'$\tilde{x}$', markersize=11)
    if show_h1 or show_h2:
        plt.legend(bbox_to_anchor=(0.32, -0.005), loc="upper left", ncol=2)
    else:
        plt.legend(bbox_to_anchor=(0.4, -0.005), loc="upper left", ncol=1)
    plt.axis("off")
    
    loc_str = "./synthetic_data"
    if show_h1:
        loc_str += "_h1.pdf"
    elif show_h2:
        loc_str += "_h2.pdf"
    else:
        loc_str += "_hstar.pdf"
        
    plt.savefig(loc_str, bbox_inches='tight')


# In[170]:


plot_synthetic(corners, reference_points)


# In[171]:


plot_synthetic(corners, reference_points, show_h1=True)


# In[172]:


plot_synthetic(corners, reference_points, show_h2=True)


# In[9]:


reference_domain = torch.zeros(4,2, device=cuda)
reference_labels = torch.ones(4, device=cuda)

for (index, p) in enumerate(reference_points):
    reference_domain[index] = torch.tensor([reference_points[index][0], reference_points[index][1]])
    
reference_labels = predict_label(h_star, reference_domain)

print(reference_domain)
print(reference_labels)


# In[10]:


num_points = 200
x_val = torch.linspace(x_start, x_end, num_points, device=cuda)
y_val = torch.linspace(y_start, y_end, num_points, device=cuda)

domain = torch.zeros(num_points**2, 2, device=cuda)

for index in range(num_points**2):
    domain[index] = torch.tensor([x_val[index//num_points], y_val[index%num_points]])

true_labels = predict_label(h_star, domain)   


# In[24]:


# find subset_domain and subset_labels
from torchmetrics.functional import pairwise_cosine_similarity


# In[89]:


a = torch.randn(1000, 784)
b = torch.randn(10, 784)

#b[0] = a[0]


# In[90]:


pairwise_cosine_similarity(a, b[:5])


# In[91]:


c = pairwise_cosine_similarity(a, b[:5])
#d = pairwise_cosine_similarity(a[100:101], b)
#pairwise_cosine_similarity(b, zero_diagonal=False)

print(c.shape)
#print(c[100])
#print(d)


# In[92]:


c


# In[93]:


e = torch.max(c, dim=1).values
e.shape


# In[94]:


e


# In[95]:


torch.argmin(e)


# In[96]:


e[torch.argmin(e).item()]


# In[97]:


torch.mean(e)


# In[62]:


b[1] = a[273]


# In[73]:


[10] + list(a.shape)[1:]


# In[70]:


torch.zeros([4,5,6]).shape


# In[84]:


torch.randint()


# In[99]:


true_labels.dtype


# In[113]:


len(torch.unique(true_labels))


# In[110]:


domain[true_labels==1].shape


# In[112]:


true_labels[true_labels==1].shape


# In[114]:


def get_greedy_subset(domain, num_points):
    inp_shape = list(domain.shape)[1:]
    subset_shape = [num_points] + inp_shape
    rand_index = torch.randint(0,len(domain), (1,)).item()
    subset_domain = torch.zeros(subset_shape, device=cuda)
    #subset_labels = torch.zeros(num_points, dtype=torch.int64, device=cuda)
    subset_domain[0] = domain[rand_index]
    
    for index in range(1, num_points):
        sim = pairwise_cosine_similarity(domain, subset_domain[:index])
        max_sim = torch.max(sim, dim=1).values
        selected_index = torch.argmin(max_sim).item()
        subset_domain[index] = domain[selected_index]
        #subset_labels[index] = labels[selected_index]
    return subset_domain #, subset_labels    


# In[119]:


def get_greedy_class_subset(domain, true_labels, num_labels, num_points):
    #num_labels = len(torch.unique(true_labels))
    
    subset_domain_class = dict()
    for label in range(num_labels):
        sub_domain = domain[true_labels == label]
        subset_domain_class[label] = get_greedy_subset(sub_domain, num_points//num_labels)
    
    return subset_domain_class    


# In[120]:


subset_domain_class = get_greedy_class_subset(domain, true_labels, 2, 100)


# In[123]:


subset_domain_class[0].shape


# In[130]:


subset_domain = get_greedy_subset(domain, 100)


# In[132]:


torch.min(torch.max(pairwise_cosine_similarity(domain, subset_domain), dim=1).values).item()


# In[107]:


subset_labels[subset_labels == 1].shape


# In[124]:


def plot_subset(corners, subset_domain_class):
    x_left, y_left, x_right, y_right, x_down, y_down, x_up, y_up = generate_data_domain(corners)
    x, y_star, y_1, y_2 = generate_decision_boundary(corners)

    # plot the data domain
    plt.plot(x_left, y_left, color="black", linewidth=3)
    plt.plot(x_right, y_right, color="black", linewidth=3)
    plt.plot(x_down, y_down, color="black", linewidth=3)
    plt.plot(x_up, y_up, color="black", linewidth=3)
    
    # plot true labeling decision boundary 
    plt.plot(x, y_star, label=r'$h^{\star}$', linewidth=3)
    plt.fill_between(x, y1=y_star, y2=y_down, color="red", alpha=0.05)
    plt.fill_between(x, y1=y_star, y2=y_up, color="green", alpha=0.05)

    for k in subset_domain_class.keys():
        subset_domain = subset_domain_class[k]
        subset_np = subset_domain.cpu().numpy()
        subset_x = [d[0] for d in subset_np]
        subset_y = [d[1] for d in subset_np]
        plt.scatter(subset_x, subset_y, label="subset_" + str(k), s=10)
    
    loc_str = "./greedy_subset_" + str(len(subset_domain)) + ".pdf"
        
    plt.savefig(loc_str, bbox_inches='tight')


# In[148]:


subset_domain_class = get_greedy_class_subset(domain, true_labels, 2, 20)
plot_subset(corners, subset_domain_class)


# In[149]:


print(torch.min(torch.max(pairwise_cosine_similarity(domain[true_labels == 0], subset_domain_class[0]), dim=1).values).item())
print(torch.min(torch.max(pairwise_cosine_similarity(domain[true_labels == 1], subset_domain_class[1]), dim=1).values).item())


# In[135]:


subset_domain_class_labels = dict()
for k in subset_domain_class.keys():
    subset_domain_class_labels[k] = [k]*len(subset_domain_class[k])


# In[11]:


def compute_unsafe_dir(reference_input, reference_label, selected_domain, selected_labels, classifier = None, beta=0.7):
    assert len(selected_domain) == len(selected_labels)
    
    unsafe_dirs = torch.zeros_like(selected_domain)
    unsafe_normalization = torch.ones(len(selected_domain), device=cuda)*float('inf')
    
    for (index, input) in enumerate(selected_domain):
        norm_diff = torch.linalg.norm(input - reference_input, ord=2)
        unsafe_dirs[index] = (input-reference_input)/norm_diff
            
        if selected_labels[index] != reference_label:
            if classifier is not None: 
                # binary search to compute normalization when the true labeling function is given
                low = 0
                high = 1
                while high - low > 1e-5:
                    alpha = (high + low)/2
                    input_alpha = reference_input + alpha*(input-reference_input)
                    label_alpha = predict_label(classifier, input_alpha)
                    if label_alpha == reference_label:
                        low = alpha
                    else:
                        high = alpha
                alpha = low
                input_alpha = reference_input + alpha*(input-reference_input)
                unsafe_normalization[index] = torch.linalg.norm(input_alpha-reference_input, ord=2)
            else:
                unsafe_normalization[index] = norm_diff*beta 
    
    return unsafe_dirs, unsafe_normalization


# In[188]:


cpu = torch.device("cpu")
cpu


# In[190]:


a = torch.randn(2,device=cuda)
print(a)
a.get_device()


# In[193]:


b = a.to(device=cpu)
print(b)
b.get_device()


# In[187]:


a


# In[179]:


a = float('inf')
a = torch.randn(2,device=cuda)
a.to_device('cpu')


# In[183]:


b = torch.randn(2)
b.get_device()


# In[194]:


def non_isotropic_dist(unsafe_dir, unsafe_normalization, perturbation, dist_type='PL', return_device='cpu', return_type=None):
    if dist_type not in ['PL', 'PD', 'WD']:
        raise ValueError('Non-isotropic distance type should be one of PL, PD, WD')
    
    distance = float('inf')
    
    element_wise_mul = torch.mul(unsafe_dir, perturbation)
    
    if dist_type == 'PL' or dist_type == 'PD':
        scaled_projections = torch.div(torch.sum(element_wise_mul, dim=1), unsafe_normalization)
        if dist_type == 'PL':
            distance = torch.max(scaled_projections)
        elif dist_type == 'PD':
            distance = torch.max(torch.abs(scaled_projections))
    
    if dist_type == 'WD':
        unsafe_4_norm = torch.linalg.norm(unsafe_dir, dim=1, ord=4)**2 # 4-norm
        modified_normalization = torch.mul(unsafe_normalization, unsafe_4_norm)
        scaled_distances = torch.div(torch.linalg.norm(element_wise_mul, dim=1, ord=2), modified_normalization)
        distance = torch.max(scaled_distances)
    
    distance = distance.detach().to(device=return_device)
    
    if return_type is not None:
        return distance.numpy()
    else:
        return distance


# In[ ]:


def non_isotropic_projection(unsafe_dir, unsafe_normalization, perturbation, epsilon=1.0, dist_type='PL', num_rounds=3):
    if dist_type not in ['PL', 'PD', 'WD']:
        raise ValueError('Non-isotropic distance type should be one of PL, PD, WD')
    
    if dist_type == 'WD':
        raise ValueError('Projection onto WD currently unsupported')
    
    device = perturbation.get_device()
    scale_flag = False 
    current_perturbation = perturbation 
    
    for t in range(len(num_rounds)):
        for (u, M) in zip(unsafe_dir, unsafe_normalization):
            if dist_type == 'PL' or dist_type == 'PD':
                distance = torch.sum(torch.mul(u, current_perturbation))/M 
                current_perturbation = current_perturbation - (distance-epsilon)*u
                if dist_type == 'PD':
                    distance = torch.sum(torch.mul(-u, current_perturbation))/M
                    current_perturbation = current_perturbation - (distance-epsilon)*(-u)
            else:
                continue 
        
        distance = non_isotropic_dist(unsafe_dir, unsafe_normalization, current_perturbation, dist_type=dist_type, return_device=device, return_type=None)
        if distance <= epsilon:
            break 
    
    distance = non_isotropic_dist(unsafe_dir, unsafe_normalization, current_perturbation, dist_type=dist_type, return_device=device, return_type=None)
    
    if distance > epsilon:
        scale_flag = True
        current_perturbation *= (epsilon/distance)
    
    return current_perturbation, scale_flag


# In[13]:


distance_types = ['l2', 'l1', 'linf', 'PL', 'PD', 'WD']


# In[14]:


def latexify_dist(dist_type: str):
    if dist_type == 'l2': return r'$\ell_2$'
    elif dist_type == 'linf': return r'$\ell_{\infty}$'
    elif dist_type == 'l1': return r'$\ell_1$'
    else: return dist_type

def latexify_inp(index: int):
    if index == 0: return r'$x$'
    elif index == 1: return r'$x_1$'
    elif index == 2: return r'$x_2$'
    elif index == 3: return r'$\tilde{x}$'
    
def latexify_h(index):
    if index is None: return r'$h^{\star}$'
    elif index == 1: return r'$h_1$'
    elif index == 2: return r'$h_2$'


# In[168]:


def get_distances(reference_input, reference_label, domain, true_labels, selected_domain = None, selected_labels = None, classifier = None, dist_type='l2'):
    print("Computing " + dist_type + " distances")
    if dist_type not in distance_types:
        raise ValueError('Distance type should be one of l2, l1, linf, PL, PD, WD')
    
    if selected_domain is not None and selected_labels is not None:
        assert len(selected_domain) == len(selected_labels)
        unsafe_dir, unsafe_normalization = compute_unsafe_dir(reference_input, reference_label, selected_domain, selected_labels, classifier=classifier)
    else:
        unsafe_dir, unsafe_normalization = compute_unsafe_dir(reference_input, reference_label, domain, true_labels, classifier=classifier)        
    
    distances = []
    for (index, input) in enumerate(domain):
        if index % 100 == 0:
            clear_output(wait=True)
            print("Computing distances at index: " + str(index) +  ", out of " + str(len(domain) - 1) + " points.")
        perturbation = input - reference_input
        if dist_type == 'l2':
            distance = torch.linalg.norm(perturbation, ord=2).detach().cpu().numpy()
        elif dist_type == 'l1':
            distance = torch.linalg.norm(perturbation, ord=1).detach().cpu().numpy()
        elif dist_type == 'linf':
            distance = torch.linalg.norm(perturbation, ord=float('inf')).detach().cpu().numpy()
        elif dist_type in ['PL', 'PD', 'WD']:
            distance = non_isotropic_dist(unsafe_dir, unsafe_normalization, perturbation, dist_type)
        distances.append(distance)
    return distances

def sublevel_set(selected_domain, distances, threshold):
    sublevel = []
    for input, distance in zip(selected_domain, distances):
        if distance <= threshold:
            sublevel.append(input.detach().cpu().numpy())
    return sublevel


# In[140]:


l_keys = subset_domain_class.keys()
print(len(l_keys))


# In[142]:


l_keys_list = list(l_keys)


# In[144]:


l_keys_list[0]


# In[143]:


subset_domain_class[l_keys_list[0]].shape


# In[145]:


for i in range(2):
    print(i)


# In[169]:


def generate_distances(reference_domain, reference_labels, domain, true_labels, classifier=h_star, distance_types=distance_types, greedy_subset=False, num_points=100):
    selected_domain = None 
    selected_labels = None 
    
    if greedy_subset:
        num_labels = len(torch.unique(true_labels))
        subset_domain_class = get_greedy_class_subset(domain, true_labels, num_labels, num_points)
        inp_shape = list(domain.shape)[1:]
        selected_domain_shape = [num_points] + inp_shape
        selected_domain = torch.zeros(selected_domain_shape, device=cuda)
        selected_labels = torch.ones(num_points, device=cuda)
        label_keys = list(subset_domain_class.keys())
        increment = num_points//num_labels
        for label_index in range(len(label_keys)):
            selected_domain[label_index*increment:(label_index+1)*increment] = subset_domain_class[label_keys[label_index]]
            selected_labels[label_index*increment:(label_index+1)*increment] *= label_keys[label_index]

    distances = dict()
    for dist_type in distance_types:
        if greedy_subset and dist_type not in ['PL', 'PD', 'WD']:
            continue 
                
        for index in range(len(reference_domain)):
            key = "x" + str(index) + "_" + dist_type
            if greedy_subset:
                key += "_greedy_" + str(num_points)
            
            distances[key] = get_distances(reference_domain[index], reference_labels[index], domain, true_labels, selected_domain=selected_domain, selected_labels=selected_labels, classifier=classifier, dist_type=dist_type)
    
    return distances, selected_domain, selected_labels        


# In[170]:


greedy_distances, selected_domain, selected_labels = generate_distances(reference_domain, reference_labels, domain, true_labels, greedy_subset=True, num_points=20)


# In[171]:


greedy_distances.keys()


# In[176]:


def plot_distance_intensity(corners, reference_points, distances, domain, selected_domain=None, distance_types=distance_types, greedy_subset=False, num_points=100):
    x0, x1, x2, xtil = reference_points[0], reference_points[1], reference_points[2], reference_points[3]
    x_left, y_left, x_right, y_right, x_down, y_down, x_up, y_up = generate_data_domain(corners)
    x, y_star, y_1, y_2 = generate_decision_boundary(corners)
    
    domain_np = domain.cpu().numpy()
    domain_x = [d[0] for d in domain_np]
    domain_y = [d[1] for d in domain_np]
    
    for dist_type in distance_types:
        if greedy_subset and dist_type not in ['PL', 'PD', 'WD']:
            continue 
        
        for index in range(len(reference_points)):
            key = "x" + str(index) + "_" + dist_type
            if greedy_subset:
                key += "_greedy_" + str(num_points)
            
            distance_domain = distances[key]
            plt.figure()
            title_str = latexify_dist(dist_type) + " threat "
            title_str += " from reference point " + latexify_inp(index)
            plt.title(title_str)
            
            # plot the data domain
            plt.plot(x_left, y_left, color="black", linewidth=3)
            plt.plot(x_right, y_right, color="black", linewidth=3)
            plt.plot(x_down, y_down, color="black", linewidth=3)
            plt.plot(x_up, y_up, color="black", linewidth=3)
    
            # plot true labeling decision boundary 
            plt.plot(x, y_star, label=r'$h^{\star}$', linewidth=3)
        
            # plot reference points
            plt.plot(x0[0], x0[1], color="black", marker=latexify_inp(0), markersize=9)
            plt.plot(x1[0], x1[1], color="black", marker=latexify_inp(1), markersize=12)
            plt.plot(x2[0], x2[1], color="black", marker=latexify_inp(2), markersize=12)
            plt.plot(xtil[0], xtil[1], color='black', marker=latexify_inp(3), markersize=12)
            
            # Plot the distance intensity map
            plt.scatter(domain_x, domain_y, c=distance_domain, s=2, cmap="plasma")
            plt.colorbar(orientation='horizontal')
            
            if selected_domain is not None:
                selected_np = selected_domain.cpu().numpy()
                selected_x = [d[0] for d in selected_np]
                selected_y = [d[1] for d in selected_np]
                plt.scatter(selected_x, selected_y, color="black", s=5)
    
            
            plt.axis('off')
            plt.savefig("./distance_intensity_" + key + ".pdf", bbox_inches='tight')


# In[177]:


plot_distance_intensity(corners, reference_points, greedy_distances, domain, selected_domain=selected_domain, greedy_subset=True, num_points=20)   


# In[197]:


# Compute certified threshold for a given point based on its distances matrix and the collection of all points. 
def compute_certified_threshold(reference_domain, reference_labels, domain, true_labels, distances):
    certified_thresholds = dict()
    for dist_type in distance_types:
        for r_index in range(len(reference_domain)):
            key = "x" + str(r_index) + "_" + dist_type
            reference_label = reference_labels[r_index]
            threshold = float('inf')
            for index in range(len(domain)):
                if true_labels[index] != reference_label and threshold > distances[key][index]:
                    threshold = distances[key][index]
            certified_thresholds[key] = threshold
    return certified_thresholds


# In[198]:


reference_labels_h1 = predict_label(h_1, reference_domain)
reference_labels_h2 = predict_label(h_2, reference_domain)

domain_labels_h1 = predict_label(h_1, domain)
domain_labels_h2 = predict_label(h_2, domain)

certified_thresholds_hstar = compute_certified_threshold(reference_domain, reference_labels, domain, true_labels, distances)
certified_thresholds_h1 = compute_certified_threshold(reference_domain, reference_labels_h1, domain, domain_labels_h1, distances)
certified_thresholds_h2 = compute_certified_threshold(reference_domain, reference_labels_h2, domain, domain_labels_h2, distances)


# In[210]:


str(certified_thresholds_hstar['x0_PL'])


# In[212]:


latexify_h(None)


# In[213]:


latexify_dist('l2')


# In[217]:


latexify_inp(0)


# In[225]:


def plot_sublevel_sets(corners, reference_points, distances, domain, certified_thresholds, distance_types=distance_types, h_index=None):
    x0, x1, x2, xtil = reference_points[0], reference_points[1], reference_points[2], reference_points[3]
    x_left, y_left, x_right, y_right, x_down, y_down, x_up, y_up = generate_data_domain(corners)
    x, y_star, y_1, y_2 = generate_decision_boundary(corners)
    h_str = "h" + str(h_index) if h_index is not None else "hstar"
            
    domain_np = domain.cpu().numpy()
    domain_x = [d[0] for d in domain_np]
    domain_y = [d[1] for d in domain_np]
    
    for dist_type in distance_types:
        for index in range(len(reference_points)):
            key = "x" + str(index) + "_" + dist_type
            distance_domain = distances[key]
            certified_threshold = certified_thresholds[key]
            cert_str = "%.2f" % certified_threshold 
            title_str = latexify_dist(dist_type) + " certified threshold for " + latexify_h(h_index) + " at " + latexify_inp(index) + " is " + cert_str
            plt.figure()
            plt.title(title_str)
            
            # plot the data domain
            plt.plot(x_left, y_left, color="black", linewidth=3)
            plt.plot(x_right, y_right, color="black", linewidth=3)
            plt.plot(x_down, y_down, color="black", linewidth=3)
            plt.plot(x_up, y_up, color="black", linewidth=3)
    
            # plot true labeling decision boundary 
            plt.plot(x, y_star, label=r'$h^{\star}$', linewidth=3)
            
            if h_index == 1:
                plt.plot(x, y_1, color='magenta', label=latexify(h_index), linestyle="dashed")
                plt.fill_between(x, y1=y_1, y2=y_star, color= "magenta", alpha= 0.05, hatch='///')
            elif h_index == 2:
                plt.plot(x, y_2, color='purple', label=latexify(h_index), linestyle="dashed")
                plt.fill_between(x, y1=y_2, y2=y_star, color= "purple", alpha= 0.05, hatch='///')
    
            # plot reference points
            plt.plot(x0[0], x0[1], color="black", marker=r"$x$", markersize=9)
            plt.plot(x1[0], x1[1], color="black", marker=r"$x_1$", markersize=12)
            plt.plot(x2[0], x2[1], color="black", marker=r"$x_2$", markersize=12)
            plt.plot(xtil[0], xtil[1], color='black', marker=r'$\tilde{x}$', markersize=12)
            
            # Plot the sublevel sets upto certified threshold
            epsilons = np.linspace(0, certified_threshold, 5)
            for eps in epsilons: 
                sublevel = sublevel_set(domain, distance_domain, eps)
                sublevel_x = [d[0] for d in sublevel]
                sublevel_y = [d[1] for d in sublevel]
                plt.scatter(sublevel_x, sublevel_y, c='red', s=0.1/(0.1+eps))

            plt.axis('off')
            plt.savefig("./certified threshold_" + h_str + key + ".pdf", bbox_inches='tight')



# In[ ]:


plot_sublevel_sets(corners, reference_points, distances, domain, certified_thresholds_hstar, h_index=None)

plot_sublevel_sets(corners, reference_points, distances, domain, certified_thresholds_h1, h_index=1)

plot_sublevel_sets(corners, reference_points, distances, domain, certified_thresholds_h2, h_index=2)
    


# In[ ]:


# unsafe_dir, unsafe_normalization = compute_unsafe_dir(p_ref, domain)
# unsafe_pert = []
# for d in unsafe_dir:
#     unsafe_pert.append(sum(p_ref, d))

# unsafe_pert_x = [d.x for d in unsafe_pert]
# unsafe_pert_y = [d.y for d in unsafe_pert]


# sublevel_PL = sublevel_set(domain, distance_domain_PL, 0.0)

# sublevel_PL_x = [d.x for d in sublevel_PL]
# sublevel_PL_y = [d.y for d in sublevel_PL]


# plt.scatter(sublevel_PL_x, sublevel_PL_y, c='green', s=0.1)
# plt.scatter(unsafe_pert_x, unsafe_pert_y, c='red', s=0.1)
# #plt.plot([3.8],[0.9], color='black', marker=r'$x$', markersize=9)
# #plt.plot([2.7],[0.361], color='red', marker=r'$x_1$', markersize=12)
# #plt.plot([1.4],[-0.276], color='red', marker=r'$\tilde{x}$', markersize=12)
# plt.plot([4.94],[0.452], color='red', marker=r'$x_2$', markersize=12)
# plt.plot(x, y, color="black", label="h*", linewidth=3)
# plt.plot(x1,y1, color="black", linewidth=3)
# plt.plot(x2,y2, color="black", linewidth=3)
# plt.plot(x3,y3, color="black", linewidth=3)
# plt.plot(x4,y4, color="black", linewidth=3)
# plt.axis('off')
# plt.title("x + Unsafe Directions")

