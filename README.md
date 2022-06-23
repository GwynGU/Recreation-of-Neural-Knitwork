# Recreation-of-Neural-Knitwork

This code was created as one of the steps in a research project Dr. Navid Kardan and I worked on over the summer as part of the University of Central Florida's CRCV REU 2022.

Our goal was to recreate the Neural Knitwork architecture found in the paper *Neural Knitworks: Patched Neural Implicit Representation Networks*. The main difference between our implementation and theirs is that we do not have a multiscale patch representation, we only have one patch scale. We also do not have cross-patch consistency or a discriminator. The code for the patch MLP and Fourier feature mapping was adapted from Jax code for the paper *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, which can be found [here.](https://github.com/tancik/fourier-feature-networks) 
