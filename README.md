# EIBoris

Explainable and Interpretable Boris is an extension of the original BorisCAD and aims to introduce more explainability concepts at the neural network level.
The original BorisCAD version is released as V1.0.0.

# TODO

### Main Branch

- [ ] Change metrics to Pytorch Ignite
  - [ ] Include metrics to provide per-class accuracy
  - [ ] Evaluate how class imbalance influences the system (we have 3 datasets, two imbalanced and one balanced and can compare using that)
- [ ] Update config.yaml parsing to support missing arguments (e.g. training without segmentation or resuming previous saved training)
  - [x] Disable segmentation (changed it to also support the same number of epochs for each model or only int epochs)
  - [ ] Resume previous training based on a saved file
- [ ] Try to add LMS weighting to the full system influence as proposed
- [ ] Train memory in X images as part of the training process
- [ ] Rework how the feature maps are accessed from within the ResNet model
- [ ] Pipeline the training process, in a way that is possible to control (as in a real situation with a radiologist)
  - [ ] Add a GUI for the training
  - [ ] Add explainable methods (e.g. final rules for the decision extracted from the DTs and the similar images in memory that were taken on the final decision)

### EIBoris Branch

- [ ] Integrate IcCNN into the current model
- [ ] Understand how to manage the training process from and to these filters
- [ ] Remove all the bloat their features have, keeping it to the minimum
- [ ] Figure out how to extract the filters and display them
- [ ] Extend the explainability provided in the main branch to also show these filter activations.

Changes to the TODO list are made in the _main_ branch whenever it is pushed.
