# jet-jitter
Python code that uses an Bayesian Markov-Chain Monte Carlo (MCMC) algorithm to model and correct for small-scale positional offsets in snapshot images of X-ray binary jets produced from the Very Long Baseline Array (VLBA) radio frequency telescope.

This code is implemented in the publication [Miller-Jones et al. 2019, Nature, XX, XX-XX]().

## Brief Description
Low level positional offsets can occur between individual snapshot images taken with the VLBA due to short timescale trosphospheric phase variations, and as a sideffect of the self-calibration process shifting the source positons by a fraction of the synthesized beam. These offsets effect our ability to track the motion of resolved jet components in images of X-ray binaries, and in turn accuratly measure proper motions and ejection times.

Assuming the motion of the jet components is ballistic, we construct a series of linear equations with *k* ejecta components and *i* snapshot images, such that,

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%5Cnonumber%20%7B%5Crm%20RA%7D_%7Bik%7D%26%3D%5Cmu_%7B%7B%5Crm%20ra%7D%2Ck%7D%28t_i-t_%7B%7B%5Crm%20ej%7D%2Ck%7D%29&plus;J_%7B%7B%5Crm%20ra%7D%2Ci%7D%2C%5C%5C%5Cnonumber%20%26%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%7B%5Crm%20and%7D%5C%5C%5Cnonumber%20%7B%5Crm%20Dec%7D_%7Bik%7D%26%3D%5Cmu_%7B%7B%5Crm%20dec%7D%2Ck%7D%28t_i-t_%7B%7B%5Crm%20ej%7D%2Ck%7D%29&plus;J_%7B%7B%5Crm%20dec%7D%2Ci%7D.%5C%5C%5Cnonumber%20%5Cend%7Balign%7D)

Here ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_%7B%7B%5Crm%20ra/dec%7D%2Ck%7D) represents the proper motions in RA/dec, and ![equation](https://latex.codecogs.com/gif.latex?t_%7B%7B%5Crm%20ej%7D%2Ck%7D) represents the ejection time of the *k*th jet component, while ![equation](https://latex.codecogs.com/gif.latex?J_%7B%7B%5Crm%20ra/dec%7D%2Ci%7D) are the jitter parameters representing an offset in position for each *i*th image.

This code uses an MCMC algorithm, implemented by the emcee package, to simultaneously solve for all of the proper motions, ejections times and jitter parameters. This code also offers the option to downweight components that have lower confidence.

## Usage
Input: 
* Data file containing jet component positions (.txt) - columns are UT time, RA offset, error in RA offset, Dec offset, error in Dec offset, flux, error in flux, component name, confidence flag.

Output:
* Diagnostic plots - histograms and trace plots of MCMC output, before and after jitter corrected positions, position angles of jet components
* Best-fit parameter file:
** Columns are best fit value, lower confidence interval error, higher confidence interval error
** Rows cycle througfh all components showing,
RA proper motion
Dec proper motion
Ejection time
Followed by a cycle through the jitter parameters for each time bin showing,
RA jitter
Dec jitter

## Requires the following python packages
* emcee
* numpy
* scipy
* matplotlib
* astropy
