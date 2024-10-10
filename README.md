# VQ-VAE-Hopfield
Relaxed version of the VQ-VAE based on modern Hopfield networks [1]. This work is an extension of the VQ-VAE paper [2] and the code provided here has been built upon [3]. 



To reproduce the results, run the scripts in the following order: 

- Imported libraries
- Hyyperparameters
- Dataset
- HopfieldQuantizer/VectorQuantizer/VectorQuantizerEMA (to use VectorQuantizerEMA, change decay in Hyperparams from 0 to 0.99)
- EncoderDecoder architecture
- ModelClass
  
- TrainingDynamics 
- Validation
- TemperatureComparisons


[1]: https://arxiv.org/abs/2008.02217
[2]: https://arxiv.org/abs/1711.00937
[3]: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
