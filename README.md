# CERTIFIED ROBUSTNESS TO DATA POISONING IN GRADIENT-BASED TRAINING

A package for abstract gradient training of neural networks for certificates of poisoning robustness.

## Important Note

This package supports certificates against the following attack models:

- Bounded adversaries: Implemented in the `poison_certified_training` method.
- Unbounded adversaries: Implemented in the `privacy_certified_training` method. Though this is labelled as `privacy`, it is equivalent to the unbounded adversary model in the paper. It is labelled as such due to its additional applications in privacy.

## Getting started

To train a `torch.nn.Sequential` model with abstract gradient training, you must set up a dataloader, model and configuration object and then call the corresponding certified training method.

```python
import torch
import abstract_gradient_training as agt
# set up dataloaders
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=1000)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
# set up pytorch model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)
# set up configuration object
config = agt.AGTConfig(
    n_epochs=10,
    learning_rate=0.1,
    k_poison=10,
    epsilon=0.01,
    loss="cross_entropy"
)
# run certified training
param_l, param_n, param_u = agt.poison_certified_training(model, config, dl_train, dl_test)
```

Additional usage examples can be found in the `examples` directory.
