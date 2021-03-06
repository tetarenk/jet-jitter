# jet-jitter
Python code that uses a Bayesian Markov-Chain Monte Carlo (MCMC) algorithm to model and correct for small-scale positional offsets in snapshot Very Long Baseline Array (VLBA) images of the jets from X-ray binary V404 Cygni.

This code is used for analysis of the data presented in the publication [Miller-Jones, Tetarenko et al. 2019, Nature, 569, 374-377](https://doi.org/10.1038/s41586-019-1152-0).

## Brief Description
Low level positional offsets can occur between individual snapshot images taken with the VLBA due to short timescale trophospheric phase variations, and as a side-effect of the self-calibration process shifting the source positons by a fraction of the synthesized beam. These offsets affect our ability to track the motions of resolved jet components in images of V404 Cygni, and in turn prevent us from accurately measuring the proper motions and ejection times of the jet components.

Assuming the motion of the jet components is ballistic, we construct a series of linear equations with *k* ejecta components and *i* snapshot images, such that,

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign%7D%5Cnonumber%20%7B%5Crm%20RA%7D_%7Bik%7D%26%3D%5Cmu_%7B%7B%5Crm%20ra%7D%2Ck%7D%28t_i-t_%7B%7B%5Crm%20ej%7D%2Ck%7D%29&plus;J_%7B%7B%5Crm%20ra%7D%2Ci%7D%2C%5C%5C%5Cnonumber%20%26%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%7B%5Crm%20and%7D%5C%5C%5Cnonumber%20%7B%5Crm%20Dec%7D_%7Bik%7D%26%3D%5Cmu_%7B%7B%5Crm%20dec%7D%2Ck%7D%28t_i-t_%7B%7B%5Crm%20ej%7D%2Ck%7D%29&plus;J_%7B%7B%5Crm%20dec%7D%2Ci%7D.%5C%5C%5Cnonumber%20%5Cend%7Balign%7D)

Here  ![equation](https://latex.codecogs.com/gif.latex?%5Cmu_%7B%7B%5Crm%20ra/dec%7D%2Ck%7D)  represents the proper motions in RA/Dec, and  ![equation](https://latex.codecogs.com/gif.latex?t_%7B%7B%5Crm%20ej%7D%2Ck%7D)  represents the ejection time of the *k*th jet component, while  ![equation](https://latex.codecogs.com/gif.latex?J_%7B%7B%5Crm%20ra/dec%7D%2Ci%7D)  are the jitter parameters representing an offset in position for each *i*th image.

This code uses an MCMC algorithm (implemented by the emcee package) to simultaneously solve this system of equations for the proper motions, ejection times and jitter parameters. Additionally, the code also offers the option to downweight components that have lower confidence.

## Usage
Input: 
* Data file containing jet component positions (bs249_uvmultifit_ptsrc_v3_flags_update.txt) - columns are UT time string, RA offset, error in RA offset, Dec offset, error in Dec offset (all in arcsec), flux, error in flux (all in Jy), component name, confidence flag (H=high confidence, M=medium confidence, L=low confidence, B=blended component, D=don't include in fit). 

Output:
* Diagnostic plots - histograms and trace plots of MCMC output, before and after jitter corrected positions versus time, corrected angular separations versus time, position angles of jet components. Here is a before and after jitter correction example (Note: N8 and N9 are not included in the fit due to their apparent non-ballistic motion),

<img src="docs/VLBA_positions_before.png" width="350" height="350" title='Before'><img src="docs/VLBA_positions_after.png" width="350" height="350" title='After'>


* Best-fit parameter file (bestp_param.txt):
  * Columns are best fit value, 1 sigma lower confidence interval error, 1 sigma upper confidence interval error
  * Rows cycle through all jet components (core,N3,S3,N1,S5,N8,S6,N2,N4,S7,N6,S2,N9) displaying the RA proper motion (mas/hr), Dec proper motion (mas/hr) and ejection time (decimal hrs) in sequence, followed by cycling through each time bin displaying the RA jitter and Dec jitter (both in mas) in sequence.

## Requires the following python packages
* emcee
* numpy
* scipy
* matplotlib
* astropy
