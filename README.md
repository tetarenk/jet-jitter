# jet-jitter
Python code that uses an Bayesian Markov-Chain Monte Carlo (MCMC) algorithm to model and correct for atmospheric jitter 
in images of X-ray binary jets produced from the Very Long Baseline Array (VLBA) radio frequency telescope.

This code is implemented in the publication Miller-Jones et al. 2019, Nature, XX, XX-XX.

## Brief Description
Low level positional offsets can occur between individual snapshot images taken with the VLBA due to short timescale trosphospheric phase vairations, and as a sideffect of the self-calibration process shifting the source positons by a fraction of the synthesized beam. These offsets effect our ability to track the motion of resolved jet components in images of X-ray binaries, and in turn accuarelty measure proper motions and ejection times.

Assuming the motion of the jet components is ballistic, we construct a series of linear equations with k ejecta components and i snapshot images, such that,

! [equation](https://latex.codecogs.com/gif.latex?%7B%5Crm%20RA%7D_%7Bik%7D%3D%5Cmu_%7B%7B%5Crm%20ra%7D%2Ck%7D%28t_i-t_%7B%7B%5Crm%20ej%7D%2Ck%7D%29&plus;J_%7B%7B%5Crm%20ra%7D%2Ci%7D)

## Requires the following python packages
* emcee
* numpy
* scipy
* matplotlib
* astropy
