# Nonisotropic Adversarial Robustness

## Threat specification functionality

```

threat_model := ProjectedDisplacement(threat_hparams(num_chunks=True), weighted=True, segmented=False)

threat_model.prepare(num_devices=5)

threat_model.evaluate(examples, labels, perturbed_examples)

threat_model.project(examples, labels, perturbed_examples)
```
