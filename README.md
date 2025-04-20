# EEL-6812: Neural Networks - Spring 2025
## Group 15: Zachary Miles, Natalie Nguyen, David Rosenfeld

## Project Title: Forest Fire Prediction Using CNNs and Transfer Learning

This repository contains the code and data used to generate the results shown in the presentation and report.  This code will run without modification, assuming that the packages listed in requirements.txt (and their dependencies) are installed locally.  Each of the three code files listed here is standalone, and does not require the other files in order to execute.

The two datasets used in this project are found in the "data" directory.

The code files are:

- initial_cnn.py: This file contains the first, simpler CNN that was developed to produce the saved weights used in the transfer learning.

- extended_cnn.py: This file has all of the contents of initial_cnn.py, plus the more complex CNN to which the saved weights were applied before training.  This code produces graphs that compare the training times, losses, and accuracies of those two models, as shown in the presentation.

- dat.py: This file builds on extended_cnn.py to implement Domain Adversarial training by adding a gradient reversal layer, domain discriminator, and domain adversarial trainer.  It loads unlabeled images from a different dataset and uses those for the adversarial training.

### Notes
This code was developed and executed against a CPU version of PyTorch.  GPU-based versions were not used or tested, and the code's reliability against those libraries is not known.

The first two files (initial_cnn.py and extended_cnn.py) use the ForestFireDataset.  The dat.py file that implements domain adversarial training uses both the ForestFireDataset (for production of the initial weights), and the FLAME dataset (for the domain adversarial training)
