# Things to do

## CodeBase

-

## Code Refactor

- Convert point class to torch vector and have functionality that tensors converts to numpy arrays (and check devices) for plotting
- Modularize existing code to print all relevant plots and save relevant figures (sublevel sets at p0,p1,p2, p-til)
- Save the data domain as a data set. with a tag of "exact partition"
- Add data-generator code that fixes and samples from grid when asked for "synthetic2D"
- Add data-generator code that does regular data-loading for "MNIST, BlockMNIST, CIFAR-x, Imagenet"
- Add dataset-hyperparams for each.

## Threat functions

- Implement isotropic functions
- For each point, compute unsafe directions and exact normalization.
- Implement non-isotropic threat functions.

## Synthetic2D

- Choose synthetic h, h1, h2 and data domain.
- Visualize data domain with points p0, p1, p2 and ptil.
- Design class-wise marginal input distribution that are tightly concentrated. Sample 500 points for each label.
- Find certified radius for each triple (h, x, d) w.r.t each d in distance metrics (N, approx-n, k-approx-N), each x in the data domain, and h in {h*, h1,h2} by searching over sublevel sets.
- Visualize as intensity maps. Show variation between h1 and h2.

## Observed and k-Observed

- Compute unsafe directions with beta normalization w.r.t all sample points. Add resulting distances as observed-PL.
- Given any data-set find k-subsets for each class using greedy approximation of cosine similarity. Add resulting distances as k-observed-PL.
- Choose beta for normalization based on the minimal norm of all observed unsafe directions that are within a kappa threshold in cosine similarity.
- Run through visualization and certification sequence for PL (if exact), observed-PL and k-observed-PL.

## Projection

- Given a reference point pref, a new point p, Find projection of p to sublevel set Td(pref, epsilon).
- Use greedy projection algorithm for each non-isotropic distance. Run T rounds and then scale the iterate if it is still not in the sublevel set.

## Augment

- N-Mixup
- N-Project
- Visualize points added according to both on the "synthetic2D" dataset.

## Attack and Train

- N-PGD : Generic PGD + projection onto C set where C is specific by a N-threat function or just a simple threat function.
- Adversarial training with N-PGD.
- Robust Accuracy evaluation.
- Learn tiny neural network on synthetic2D. show N-PGD attack points.

## SOTA Evaluation

- Import model weights from RobustBench and plug into existing code-base.
- Compute robust accuracy under PGD and N-PGD attacks. Show ranking and robust accuracy curves.
- Finetune top-5 models with N-Mixup, N-Project and N-adversarial training, show robust accuracy on all combinations and hyperparameter sweeps.

## Certified Robustness  

- Search over approximation of predictor intensity based on LirPA, plot histograms and certified accuracy curves.
