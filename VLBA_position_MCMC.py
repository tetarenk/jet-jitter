#############################################
# jet-jitter for V404 Cygni
#############################################
'''Python code that uses an Bayesian Markov-Chain Monte Carlo (MCMC) algorithm 
to model and correct for small-scale positional offsets in snapshot images of the jets in
the X-ray binary V404 Cygni produced from the Very Long Baseline Array (VLBA) radio frequency
telescope. 

INPUT: Jet components position data file;
       FORMAT - 9 columns; start time in UT (e.g., 140941 is 14:09:41 UT),
       RA RAerr Dec Decerr offsets from reference position in arcsec, flux,
       error in flux (in Jy), component name, confidence flag (H=high confidence,
       M=medium confidence, L=low confidence, B=blended component, D=dont include in fit).

OUTPUT: (1) Diagnostic plots:
            (a) histograms and trace plots of MCMC output
            (b) before and after jitter corrected positions versus time
            (c) corrected angular separation versus time
            (d) position angles of jet components over time.
        (2) Best-fit parameter file:
        FORMAT - 3 columns; value, lower error, upper error
        Rows go as:
        for k in jet component:
            Ra proper motion (mas/hr)
            Dec proper motion (mas/hr)
            ejection time (decimal hrs)
        for i in time-bin:
            RA jitter (mas)
            Dec jitter (mas)
NOTE: All outputs are placed in the data products directory set below.

Written by: Alex J. Tetarenko
Last Updated: December 2017
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math as m
import scipy.stats as ss
import emcee
from astropy.time import Time
from astropy.io import ascii
import matplotlib.dates as mdates
import datetime as datetime
from matplotlib.ticker import AutoMinorLocator
from astropy.coordinates import SkyCoord
import itertools
from astropy.utils.console import ProgressBar
import os


def make_data_files(filename,dira):
	data = ascii.read(filename, delimiter=' ',data_start=0,names=['UT','RA','RAerr','Dec','Decerr','Flux','Fluxerr','Compnum','Flag'],guess=False)
	times=[]
	ra=[]
	dec=[]
	raerr=[]
	decerr=[]
	flux=[]
	fluxerr=[]
	flag=[]
	comp=[]
	for i in range(0,len(data['UT'])):
		ts0=data['UT']
		ts=ts0.astype('str')
		tim_s=ts[i]
        #write UT string to decimal hours and convert start times to times at the middle of the interval
		times.append(float(tim_s[0]+tim_s[1])+float(tim_s[2]+tim_s[3])/60.+(float(tim_s[4]+tim_s[5])+35.)/3600.)
        #put all offsets in mas
		ra.append(data['RA'][i]*1e3)
		raerr.append(data['RAerr'][i]*1e3)
		dec.append(data['Dec'][i]*1e3)
		decerr.append(data['Decerr'][i]*1e3)
		flux.append(data['Flux'][i])
		fluxerr.append(data['Fluxerr'][i])
		comp.append(data['Compnum'][i])
		if data['Flag'][i]=='H':#high confidence
			flag.append(1.)
		elif data['Flag'][i]=='M':#medium confidence
			flag.append(0.7)
		elif data['Flag'][i]=='L':#low confidence
			flag.append(0.3)
		elif data['Flag'][i]=='B':#blended component
			flag.append(0.1)
		elif data['Flag'][i]=='D':#dont fit do to non-ballistic motion (i.e., N2,N6,first 5 points for N4)
			flag.append(0.0)
		else:
			flag.append(0.)
    #array of unique time bins
	times_unique=np.unique(times)
    #write out individual data files for each component
	fileC=open(dira+'comps_C.txt','w')
	fileN=open(dira+'comps_N.txt','w')
	fileN1=open(dira+'comps_N1.txt','w')
	fileN2=open(dira+'comps_N2.txt','w')
	fileN3=open(dira+'comps_N3.txt','w')
	fileN4=open(dira+'comps_N4.txt','w')
	fileN5=open(dira+'comps_N5.txt','w')
	fileN6=open(dira+'comps_N6.txt','w')
	fileS=open(dira+'comps_S.txt','w')
	fileS1=open(dira+'comps_S1.txt','w')
	fileS2=open(dira+'comps_S2.txt','w')
	fileS4=open(dira+'comps_S4.txt','w')
	fileS5=open(dira+'comps_S5.txt','w')
	for i in range(0,len(times)):
		if comp[i]=='C':
			fileC.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N':
			fileN.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N1':
			fileN1.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N2':
			fileN2.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N3':
			fileN3.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N4':
			fileN4.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N5':
			fileN5.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='N6':
			fileN6.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='S':
			fileS.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='S1':
			fileS1.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='S2':
			fileS2.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='S4':
			fileS4.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
		if comp[i]=='S5':
			fileS5.write('{0} {1} {2} {3} {4} {5}\n'.format(times[i],ra[i],raerr[i],dec[i],decerr[i],flag[i]))
	fileC.close()
	fileN.close()
	fileN1.close()
	fileN2.close()
	fileN3.close()
	fileN4.close()
	fileN5.close()
	fileN6.close()
	fileS.close()
	fileS1.close()
	fileS2.close()
	fileS4.close()
	fileS5.close()
	return(times_unique)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    '''Check if value is equivalent to another

    a: val1
    b: val2

    returns: True or False
    '''
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def ra_form(time,mu_ra,t_ej,ra_off):
    '''RA position vs time

    mu_ra: proper motion in ra, mas/hr
    time: time value or array in hr on observation day
    t_ej: ejection time in hr on observation day
    ra_off: offset jitter in RA

    returns: RA position after (time-t_ej) has passed.
    '''
    return(mu_ra*(time-t_ej)+ra_off)

def dec_form(time,mu_dec,t_ej,dec_off):
    '''Dec position vs time

    mu_dec: proper motion in dec, mas/hr
    time: time value or array in hr on observation day
    t_ej: ejection time in hr on observation day
    dec_off: offset jitter in Dec

    returns: Dec position after (time-t_ej) has passed.
    '''
    return(mu_dec*(time-t_ej)+dec_off)

def bulk_motion_model(par,time):
	'''Position vs time no jitter.

	par: [mu_ra,mu_dec,t_ej]
	time: time value or array in hr on observation day

	returns: Position after (time-t_ej) has passed.
	'''
	mu_ra=par[0]
	mu_dec=par[1]
	t_ej=par[2]
	ra=mu_ra*(time-t_ej)
	dec=mu_dec*(time-t_ej)
	return(ra,dec)

def jitter_model(par,time,time_comp,num,ncomp):
    '''Jitter component positions model.

    par: parameter array
    time: array of unique time bins for which at least one component is detected.
    time_comp: array of time bins for the component in question.
    num: index of proper motion parameters for component in the parameter array 
    ncomp: number of components you have to model

    NOTE: If ncomp=1, then time and time_comp are same.

    returns: RA and Dec positions for time array based on proper motions and jitter values'''
    numc=ncomp-1
    mu_ra=par[num*3]
    mu_dec=par[num*3+1]
    t_ej=par[num*3+2]
    RA=[]
    DEC=[]
    #for each unique time bin check if component in question is detected, if so add to model
    for kk in np.arange(len(time)):
        ra_jit=par[3*numc+3+2*kk]
        dec_jit=par[3*numc+3+2*kk+1]
        match=[]
        for ll in np.arange(len(time_comp)):
            match.append(isclose(time[kk], time_comp[ll]))
        if np.any(match):
            RA.append(ra_form(time[kk],mu_ra,t_ej,ra_jit))
            DEC.append(dec_form(time[kk],mu_dec,t_ej,dec_jit))
    return(RA,DEC)

def comp_fix(datafile,bestfit,ncomp,times_unique):
    '''Apply jitter correction to data.

    datafile: data file to be corrected (time in decimal hrs, ra/dec in decimal degrees)
    bestfit: best fit parameter data file
    ncomp: number of components
    times_unique: array of unique time bins for which at least one component is detected.

    returns: fixed RA and Dec positions of a component 
     '''
    best=np.loadtxt(bestfit)
    bestp_final=best[:,0]
    jitter=(bestp_final[3*ncomp:])
    fixed_comp_ra=[]
    fixed_comp_dec=[]
    data=np.loadtxt(datafile)
    for i in range(0,len(times_unique)):
        match=[]
        for kk in np.arange(len(data[:,0])):
            match.append(isclose(times_unique[i],data[kk][0]))
        if np.any(match):
            ind=[j for j, x in enumerate(match) if x]
            fixed_comp_ra.append(((data[ind,1]))-jitter[i*2])
            fixed_comp_dec.append(((data[ind,3]))-jitter[i*2+1])
    return(fixed_comp_ra,fixed_comp_dec)

def hrs_to_mjd(VLBA_hrs):
    '''Convert time in decimal hrs to mjd

    VLBA_hrs: time array in decimal hrs

    returns: time array in mjd '''
    st=[]
    for i in VLBA_hrs:
        hrs=int(m.modf(i)[1])
        minn=m.modf(i)[0]*60.
        secc=m.modf(minn)[0]*60.
        s='2015-06-22'+' '+str(hrs)+':'+str(int(np.floor(minn)))+':'+str(secc)
        sta=Time(s, format='iso', scale='utc').mjd
        st.append(sta)
    return(np.array(st))

def lp_flag(param,time0,Timea,Data,Error,FLAGs,fixp,guess,tinit,ncomp):
    '''Log probability function for jitter model.
    param: parameter array
    time0: array of unique time bins for which at least one component is detected.
    Timea: list of time arrays for each component to be fit; core must be first entry
    Data: list of position arrays for each component to be fit; core must be first entry
    Error: list of position error arrays for each component to be fit; core must be first entry
    FLAGs: confidence weights 
    fixp: fixed param array (True for fixed, False for free)
    guess: initial guess for all parameters
    tinit: array of guesses for ejection times for each component
    ncomp: number of components you have to model

    returns: log probability for emcee sampler
    '''
    re=[]
    for i in range(0,len(Timea)):
        if i !=0:
        	re.append((FLAGs[i]*(jitter_model(param,time0,Timea[i],i,ncomp)[0]-Data[2*i])**2/(2*Error[2*i]**2)))
        	re.append((FLAGs[i]*(jitter_model(param,time0,Timea[i],i,ncomp)[1]-Data[2*i+1])**2/(2*Error[2*i+1]**2)))
    chi2_tot=np.nansum(np.concatenate((re)))
    prior = model_prior(param,fixp,guess,tinit,ncomp,time0)
    return(-chi2_tot+prior)

def model_prior(pval,fixp,guess,tinit,ncomp,time0):
    '''Prior function for jitter model.

    pval: parameter array
    fixp: fixed param array (True for fixed, False for free)
    guess: initial guess for all parameters
    tinit: array of guesses for ejection times for each component
    ncomp: number of components you have to model
    time0: array of unique time bins for which at least one component is detected

    returns: log prior for emcee sampler
     '''
    nparam=ncomp-1
    ps=[]
    for i in range(0,len(pval)):
        if fixp[i]==True:
            ps.append(guess[i])
        elif fixp[i]==False:
            ps.append(pval[i])
        else:
            raise ValueError("The fixed parameter array values can only be True or False")
    p=np.array(ps)
    for ii in np.arange(ncomp):
        mu_ra=p[3*ii]
        mu_dec=p[3*ii+1]
        t_ej=p[3*ii+2]
        prior = 0.0
        prior += ss.norm.logpdf(mu_ra,scale=0.3,loc=guess[3*ii]) + ss.uniform.logpdf(mu_ra,loc=-2.0,scale=4.0)
        prior += ss.norm.logpdf(mu_dec,scale=0.3,loc=guess[3*ii+1]) + ss.uniform.logpdf(mu_dec,loc=-2.0,scale=4.0)
        if tinit[ii]:
            prior+= ss.uniform.logpdf(t_ej,loc=(tinit[ii]-1.0),scale=2.)
        for kk in np.arange(len(time0)):
            ra_jit=p[3*nparam+3+2*kk]
            dec_jit=p[3*nparam+3+1+2*kk]
            prior += ss.norm.logpdf(ra_jit,scale=0.5,loc=guess[3*nparam+3+2*kk]) + ss.uniform.logpdf(ra_jit,loc=-3.,scale=6.)
            prior += ss.norm.logpdf(dec_jit,scale=0.5,loc=guess[3*nparam+3+1+2*kk]) + ss.uniform.logpdf(dec_jit,loc=-3.,scale=6.)
    if np.isnan(prior):
    	return(-np.inf)
    return(prior)

def confidenceInterval(y,sig):
    '''Calculates the Gaussian sigma confidence interval for a pdf

    y: pdf in array format
    sig: sigma confidence interval

    returns: list [median,lower error bar, upper error bar]
     '''
    median=np.median(y)
    pct15=np.percentile(y,15)
    pct85=np.percentile(y,85)
    list1=np.array([median,median-pct15,pct85-median])
    return list1

####################################
####################################


#path to input data file directory- include trailing /!!
path_data='/export/data2/atetarenko/jitter_package/github/data/'
if not os.path.isdir(path_data):
    raise Exception('Please create the '+path_data+' directory and put your data files in it.')


#path to data products directory- include trailing /!!
path_dir='/export/data2/atetarenko/jitter_package/github/results/'
if not os.path.isdir(path_dir):
    os.system('mkdir '+path_dir)
print 'All data products will be saved in '+path_dir


#read in component position data files
#FORMAT - start time in UT (e.g., 140941 is 14:09:41 UT), RA RAerr Dec Decerr offsets from reference position in arcsec
print 'Reading in data files...'
times_unique=make_data_files(path_data+'bs249_uvmultifit_ptsrc_v3_flags_update.txt',path_data)
core=np.loadtxt(path_data+'comps_C.txt')
N=np.loadtxt(path_data+'comps_N.txt')
N1=np.loadtxt(path_data+'comps_N1.txt')
N2=np.loadtxt(path_data+'comps_N2.txt')
N3=np.loadtxt(path_data+'comps_N3.txt')
N4=np.loadtxt(path_data+'comps_N4.txt')
N5=np.loadtxt(path_data+'comps_N5.txt')
N6=np.loadtxt(path_data+'comps_N6.txt')
S=np.loadtxt(path_data+'comps_S.txt')
S1=np.loadtxt(path_data+'comps_S1.txt')
S2=np.loadtxt(path_data+'comps_S2.txt')
S4=np.loadtxt(path_data+'comps_S4.txt')
S5=np.loadtxt(path_data+'comps_S5.txt')


#make lists of all components to consider in fit
Timea=[core[:,0],N[:,0],S[:,0],N1[:,0],S1[:,0],N2[:,0],S2[:,0],N3[:,0],N4[:,0],S4[:,0],N5[:,0],S5[:,0],N6[:,0]]
Data=[core[:,1],core[:,3],N[:,1],N[:,3],S[:,1],S[:,3],N1[:,1],N1[:,3],S1[:,1],S1[:,3],N2[:,1],N2[:,3],S2[:,1],S2[:,3],\
N3[:,1],N3[:,3],N4[:,1],N4[:,3],S4[:,1],S4[:,3],N5[:,1],N5[:,3],S5[:,1],S5[:,3],N6[:,1],N6[:,3]]
Error=[core[:,2],core[:,4],N[:,2],N[:,4],S[:,2],S[:,4],N1[:,2],N1[:,4],S1[:,2],S1[:,4],N2[:,2],N2[:,4],S2[:,2],S2[:,4],\
N3[:,2],N3[:,4],N4[:,2],N4[:,4],S4[:,2],S4[:,4],N5[:,2],N5[:,4],S5[:,2],S5[:,4],N6[:,2],N6[:,4]]
FLAGs=[core[:,5],N[:,5],S[:,5],N1[:,5],S1[:,5],N2[:,5],S2[:,5],N3[:,5],N4[:,5],S4[:,5],N5[:,5],S5[:,5],N6[:,5]]



#set RA/Dec initial jitter params (in mas) to the core offset over time
print 'Setting initial guesses for jitter params...'
jitter_ra=core[:,1]
jitter_dec=core[:,3]

iters = [iter(jitter_ra), iter(jitter_dec)]
jitt=list(it.next() for it in itertools.cycle(iters))

#initial params guess for proper motions and ejection times
#[mu_ra (mas/h), mu_dec (mas/h),tej (decimal hours)]
print 'Setting initial guesses for component proper motions and ejection times...'
pcore=[0.0 ,0.0, 0.0]#core is assumed to be stationary, and will not be included in fit
pn=[-0.15, 0.70 ,10.9]
ps=[0.097, -0.46, 11.1]
pn1=[-0.26, 1.03, 7.0]
ps1=[0.44, -1.83 ,11.8]
pn2=[0.0,0.0,0.0]#not included in fit due to non-ballistic motion, so set to zero
ps2=[0.12, -0.26, 12.4]
pn3=[0.002, 0.79, 10.6]
pn4=[-0.07, 0.18, 11.7]
ps4=[-0.0036 ,-1.42, 12.5]
pn5=[-0.23, 0.42 ,11.1]
ps5=[0.00037 ,-0.40 ,10.4]
pn6=[0.0,0.0,0.0]#not included in fit due to non-ballistic motion, so set to zero


#initial guess array
print 'Configuring initial guess array...'
guess=pcore+pn+ps+pn1+ps1+pn2+ps2+pn3+pn4+ps4+pn5+ps5+pn6+jitt
tinitlst=[]
for kk in range(0,len(Timea)):
    tinitlst.append(guess[3*kk+2])
tinit=np.array(tinitlst)

#plot initial guess over data
print 'Plotting initial guess over data...'
ncomp=len(Timea)
ncompplot=ncomp-1
fig=plt.figure(figsize=(19,25.5))
for ii in range(0,ncompplot):
	model_ra=jitter_model([guess[3*ii+3],guess[3*ii+1+3],guess[3*ii+2+3]]+jitt,times_unique,times_unique,0,1)[0]
	model_dec=jitter_model([guess[3*ii+3],guess[3*ii+1+3],guess[3*ii+2+3]]+jitt,times_unique,times_unique,0,1)[1]
	ax=fig.add_subplot(ncompplot,2,2*ii+1)
	ax.plot(Timea[ii+1],Data[2*ii+2],marker='o')
	ax.plot(times_unique,model_ra,marker='o',ls='',color='r')
	ax=fig.add_subplot(ncompplot,2,2*ii+2)
	ax.plot(Timea[ii+1],Data[2*ii+1+2],marker='o')
	ax.plot(times_unique,model_dec,marker='o',ls='',color='r')
plt.savefig(path_dir+'comp_initial.png')
print 'Plot saved to '+path_dir+'comp_initial.png'


core_model_ra=jitter_model([guess[0],guess[1],guess[2]]+jitt,times_unique,times_unique,0,1)[0]
core_model_dec=jitter_model([guess[0],guess[1],guess[2]]+jitt,times_unique,times_unique,0,1)[1]
fig=plt.figure(figsize=(19,15.5))
ax1=fig.add_subplot(1,2,1)
ax1.plot(Timea[0],Data[0],marker='o',color='m')
ax1.plot(times_unique,core_model_ra,marker='o',ls='',color='r')
ax2=fig.add_subplot(122,sharex=ax1)
ax2.plot(Timea[0],Data[1],marker='o',color='m')
ax2.plot(times_unique,core_model_dec,marker='o',ls='',color='r')
plt.savefig(path_dir+'core_initial.png')
print 'Plot saved to '+path_dir+'core_initial.png'


#setting up sampler
print 'Setting up sampler...'
ncomp=len(Timea)
nparam=2*len(times_unique)+(3*ncomp)
nwalkers = nparam*2
nBurn = 500
nSteps = 20000
print 'The number of components: ',ncomp
print 'The number of free parameters: ',nparam
print 'Number of walkers used: ',nwalkers

print 'Defining initial position of walkers...'
fixp=np.zeros(len(guess), dtype=bool)

#we will not include the core, N2, or N6 in the fit, so we fix their paramters at 0.0 here.
fixp[0]=True
fixp[1]=True
fixp[2]=True
fixp[15]=True
fixp[16]=True
fixp[17]=True
fixp[36]=True
fixp[37]=True
fixp[38]=True

p0 = np.zeros((nwalkers,nparam))
for i in np.arange(3*ncomp):
    if fixp[i]==True:
    	p0[:,i]=guess[i]
    elif fixp[i]==False:
    	p0[:,i]=((np.random.randn(nwalkers))*0.01)+guess[i]
for idx,thisguess in enumerate(guess[3*ncomp:]):
    if fixp[idx+3*ncomp]==True:
    	p0[:,3*ncomp+idx] = thisguess
    elif fixp[idx+3*ncomp]==False:
    	p0[:,3*ncomp+idx] = (np.random.randn(nwalkers)*0.01+1.0)*thisguess

#start sampler
print 'Starting sampler...'
sampler = emcee.EnsembleSampler(nwalkers,nparam,lp_flag,args=[times_unique,Timea,Data,Error,FLAGs,fixp,guess,tinit,ncomp],\
  threads=8)#using 8 cores

print 'Performing "Burn in" sampling for', nBurn, ' steps...'
with ProgressBar(nBurn) as bar:
    for i, result in enumerate(sampler.sample(p0,iterations=nBurn)):
        bar.update()
pos,prob,state=result[0],result[1],result[2]
sampler.reset()
print 'Performing sampling for', nSteps, ' steps...'
with ProgressBar(nSteps) as bar:
    for i, result in enumerate(sampler.sample(pos,iterations=nSteps)):
        bar.update()
pos,prob,state=result[0],result[1],result[2]

#save best fit params to file
print 'Saving best-fit parameters to file...'
bestp_file=open(path_dir+'bestp_param.txt','w')
for i in range(0,int(nparam)):
    a=confidenceInterval(sampler.flatchain[:,i],1)
    bestp_file.write('{0} {1} {2}\n'.format(a[0],a[1],a[2]))
    print confidenceInterval(sampler.flatchain[:,i],1)
bestp_file.close()
bestp_final = np.median(sampler.flatchain,axis=0)

#save final walker positions
print 'Saving final walker positions to '+path_dir+'pos.txt...'
np.savetxt(path_dir+'pos.txt',pos)
#save pdfs (i.e., flatchains) for all params
print 'Saving flatchains to '+path_dir+'chains.txt...'
np.savetxt(path_dir+'chains.txt',sampler.flatchain)

print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

#plot final bestfit model
print 'Plotting best-fit model on top of data...'
bestp_final_list=bestp_final.tolist()
jitt_best=bestp_final_list[3*ncomp:]
ncomp=len(Timea)
ncompplot=ncomp-1
fig=plt.figure(figsize=(19,25.5))
for ii in range(0,ncompplot):
	model_ra=jitter_model([bestp_final[3*ii+3],bestp_final[3*ii+1+3],bestp_final[3*ii+2+3]]+jitt_best,times_unique,times_unique,0,1)[0]
	model_dec=jitter_model([bestp_final[3*ii+3],bestp_final[3*ii+1+3],bestp_final[3*ii+2+3]]+jitt_best,times_unique,times_unique,0,1)[1]
	ax=fig.add_subplot(ncompplot,2,2*ii+1)
	ax.plot(Timea[ii+1],Data[2*ii+2],marker='o')
	ax.plot(times_unique,model_ra,marker='o',ls='',color='r')
	ax=fig.add_subplot(ncompplot,2,2*ii+2)
	ax.plot(Timea[ii+1],Data[2*ii+1+2],marker='o')
	ax.plot(times_unique,model_dec,marker='o',ls='',color='r')
plt.savefig(path_dir+'comp_final.png')
print 'Plot saved to '+path_dir+'comp_final.png'

core_model_ra=jitter_model([bestp_final[0],bestp_final[1],bestp_final[2]]+jitt_best,times_unique,\
  times_unique,0,1)[0]
core_model_dec=jitter_model([bestp_final[0],bestp_final[1],bestp_final[2]]+jitt_best,times_unique,\
  times_unique,0,1)[1]
fig=plt.figure(figsize=(19,15.5))
ax1=fig.add_subplot(1,2,1)
ax1.plot(Timea[0],Data[0],marker='o',color='m')
ax1.plot(times_unique,core_model_ra,marker='o',ls='',color='r')
ax2=fig.add_subplot(122,sharex=ax1)
ax2.plot(Timea[0],Data[1],marker='o',color='m')
ax2.plot(times_unique,core_model_dec,marker='o',ls='',color='r')
plt.savefig(path_dir+'core_final.png')
print 'Plot saved to '+path_dir+'core_final.png'

#diagnostic plots
print 'Making diagnostic plots...'
figa = plt.figure(figsize=(22,15.5))
for i in range(0,ncomp*3):
    plt.subplot(ncomp,3,i+1)
    patches = plt.hist(sampler.flatchain[:,i],bins=100)
figa.subplots_adjust(hspace=.5)
plt.savefig(path_dir+'hist_mu_tej.png')

figb = plt.figure(figsize=(22,15.5))
for i in range(0,ncomp*3):
    plt.subplot(ncomp,3,i+1)
    plt.plot(sampler.chain[:,:,i].T)
figb.subplots_adjust(hspace=.5)
plt.savefig(path_dir+'trace_mu_tej.png')

print 'Trace plots and histograms saved in '+path_dir

print 'Plotting positions before correction...'
fig=plt.figure(figsize=(10,10))
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2,sharex=ax1)
cm = plt.cm.get_cmap('jet',500)
coll=matplotlib.colors.Normalize(vmin=0,vmax=len(Timea))
col=matplotlib.cm.get_cmap('jet',500)
print col(coll(ii))
for ii in range(0,len(Timea)):
    if ii==0:
        ax1.errorbar(Timea[ii],Data[2*ii],yerr=Error[2*ii],marker='o',color='k',ls='')
        ax2.errorbar(Timea[ii],Data[2*ii+1],yerr=Error[2*ii+1],marker='o',color='k',ls='')
    else:
        ax1.errorbar(Timea[ii],Data[2*ii],yerr=Error[2*ii],marker='o',color=col(coll(ii)),ls='')
        ax2.errorbar(Timea[ii],Data[2*ii+1],yerr=Error[2*ii+1],marker='o',color=col(coll(ii)),ls='')
ax1.text(12.5,-0.5,'N')
ax1.text(12,0.1,'S')
ax1.text(12,-1.5,'N1')
ax1.text(13,0.7,'S1')
ax1.text(13.1,-0.4,'N2')
ax1.text(14,0.3,'S2')
ax1.text(13.8,-0.3,'N3')
ax1.text(14.3,-1,'N5')
ax1.text(13.3,0.2,'S4')
ax1.text(14.3,-0.3,'N4')
ax1.text(11.5,0.1,'S5')
ax1.text(13.2,-0.6,'N6')
#
ax2.text(12.5,2,'N')
ax2.text(12,-1,'S')
ax2.text(12.5,6,'N1')
ax2.text(13,-1.6,'S1')
ax2.text(13.2,2,'N2')
ax2.text(14,-2,'S2')
ax2.text(14,3,'N3')
ax2.text(14.3,0.8,'N5')
ax2.text(13.3,-1.8,'S4')
ax2.text(14.1,0.,'N4')
ax2.text(11.5,-1,'S5')
ax2.text(13.5,1,'N6')
ax1.set_ylim(-2.5,1)
ax2.set_ylim(-4.,8)
ax1.set_ylabel('RA offset (mas)',fontsize=15)
ax2.set_ylabel('Dec offset (mas)',fontsize=15)
ax2.set_xlabel('Time on 22/06/2016 (hrs)',fontsize=15)
ax1.tick_params(axis='both',which='minor',length=3,width=1)
ax1.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
ax2.tick_params(axis='both',which='minor',length=3,width=1)
ax2.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
plt.savefig(path_dir+'VLBA_positions_before.png',bbox_inches='tight')
print 'VLBA positions before plot saved in '+path_dir

print 'Plotting positions after correction...'
core_off_raa=comp_fix(path_data+'comps_C.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
core_off_deca=comp_fix(path_data+'comps_C.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N_off_raa=comp_fix(path_data+'comps_N.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N_off_deca=comp_fix(path_data+'comps_N.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N1_off_raa=comp_fix(path_data+'comps_N1.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N1_off_deca=comp_fix(path_data+'comps_N1.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N2_off_raa=comp_fix(path_data+'comps_N2.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N2_off_deca=comp_fix(path_data+'comps_N2.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N3_off_raa=comp_fix(path_data+'comps_N3.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N3_off_deca=comp_fix(path_data+'comps_N3.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N4_off_raa=comp_fix(path_data+'comps_N4.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N4_off_deca=comp_fix(path_data+'comps_N4.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N5_off_raa=comp_fix(path_data+'comps_N5.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N5_off_deca=comp_fix(path_data+'comps_N5.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
N6_off_raa=comp_fix(path_data+'comps_N6.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
N6_off_deca=comp_fix(path_data+'comps_N6.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
S_off_raa=comp_fix(path_data+'comps_S.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
S_off_deca=comp_fix(path_data+'comps_S.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
S1_off_raa=comp_fix(path_data+'comps_S1.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
S1_off_deca=comp_fix(path_data+'comps_S1.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
S2_off_raa=comp_fix(path_data+'comps_S2.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
S2_off_deca=comp_fix(path_data+'comps_S2.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]
S4_off_raa=comp_fix(path_data+'comps_S4.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
S4_off_deca=comp_fix(path_data+'comps_S4.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]    
S5_off_raa=comp_fix(path_data+'comps_S5.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[0]
S5_off_deca=comp_fix(path_data+'comps_S5.txt',path_dir+'bestp_param.txt',ncomp,times_unique)[1]

Data2=[core_off_raa, core_off_deca, N_off_raa ,N_off_deca,S_off_raa, S_off_deca,N1_off_raa,N1_off_deca,S1_off_raa,\
S1_off_deca,N2_off_raa,N2_off_deca,S2_off_raa,S2_off_deca,N3_off_raa,N3_off_deca,N4_off_raa,N4_off_deca,\
S4_off_raa,S4_off_deca,N5_off_raa,N5_off_deca,S5_off_raa,S5_off_deca,N6_off_raa,N6_off_deca]

fig=plt.figure(figsize=(10,10))
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2,sharex=ax1)
cm = plt.cm.get_cmap('jet',500)
coll=matplotlib.colors.Normalize(vmin=0,vmax=len(Timea))
col=matplotlib.cm.get_cmap('jet',500)
for ii in range(0,ncomp):
    model_ra=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,\
      times_unique)[0]
    model_dec=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,\
      times_unique)[1]
    if ii==0:
        ax1.errorbar(Timea[ii],np.concatenate(Data2[2*ii]),yerr=Error[2*ii],marker='o',color='k',ls='')
        ax2.errorbar(Timea[ii],np.concatenate(Data2[2*ii+1]),yerr=Error[2*ii+1],marker='o',color='k',ls='')
        ax1.plot(times_unique,model_ra,marker='',ls='-',color='k')
        ax2.plot(times_unique,model_dec,marker='',ls='-',color='k')
    else:
        ax1.errorbar(Timea[ii],np.concatenate(Data2[2*ii]),yerr=Error[2*ii],marker='o',color=col(coll(ii)),ls='')
        ax2.errorbar(Timea[ii],np.concatenate(Data2[2*ii+1]),yerr=Error[2*ii+1],marker='o',color=col(coll(ii)),ls='')
        ax1.plot(times_unique,model_ra,marker='',ls='-',color=col(coll(ii)))
        ax2.plot(times_unique,model_dec,marker='',ls='-',color=col(coll(ii)))

ax1.text(12.5,-0.5,'N')
ax1.text(12,0.1,'S')
ax1.text(12,-1.5,'N1')
ax1.text(13,0.7,'S1')
ax1.text(13.1,-0.4,'N2')
ax1.text(14,0.3,'S2')
ax1.text(13.8,-0.3,'N3')
ax1.text(14.3,-1,'N5')
ax1.text(13.3,0.2,'S4')
ax1.text(14.3,-0.3,'N4')
ax1.text(11.5,0.1,'S5')
ax1.text(13.2,-0.6,'N6')
#
ax2.text(12.5,2,'N')
ax2.text(12,-1,'S')
ax2.text(12.5,6,'N1')
ax2.text(13,-1.6,'S1')
ax2.text(13.2,2,'N2')
ax2.text(14,-2,'S2')
ax2.text(14,3,'N3')
ax2.text(14.3,0.8,'N5')
ax2.text(13.3,-1.8,'S4')
ax2.text(14.1,0.,'N4')
ax2.text(11.5,-1,'S5')
ax2.text(13.5,1,'N6')
ax1.set_ylim(-2.5,1)
ax2.set_ylim(-4.,8)
ax1.set_ylabel('RA offset (mas)',fontsize=15)
ax2.set_ylabel('Dec offset (mas)',fontsize=15)
ax2.set_xlabel('Time on 22/06/2016 (hrs)',fontsize=15)
ax1.tick_params(axis='both',which='minor',length=3,width=1)
ax1.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
ax2.tick_params(axis='both',which='minor',length=3,width=1)
ax2.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
plt.savefig(path_dir+'VLBA_positions_after.png',bbox_inches='tight')
print 'VLBA positions after plot saved in '+path_dir

print 'Plotting corrected ang separation vs time...'
fig=plt.figure(figsize=(15,10))
ax1=fig.add_subplot(1,1,1)
cm = plt.cm.get_cmap('jet',500)
coll=matplotlib.colors.Normalize(vmin=0,vmax=len(Timea))
col=matplotlib.cm.get_cmap('jet',500)
for ii in range(0,len(Timea)):
    if ii !=0:
        if bestp_final[3*ii+1]<0:#south comp
            model_ra=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,np.arange(bestp_final[3*ii+2],times_unique[-1],0.1))[0]
            model_dec=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,np.arange(bestp_final[3*ii+2],times_unique[-1],0.1))[1]
            ax1.errorbar(hrs_to_mjd(Timea[ii]),-1.*np.sqrt((np.concatenate(Data2[2*ii]))**2+(np.concatenate(Data2[2*ii+1]))**2),\
                marker='o',color=col(coll(ii)),ls='',ms=10)
            ax1.plot(hrs_to_mjd(np.arange(bestp_final[3*ii+2],times_unique[-1],0.1)),-1.*np.sqrt(np.array(model_ra)**2+np.array(model_dec)**2),marker='',ls='-',color=col(coll(ii)))
        else:#north comp
            model_ra=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,np.arange(bestp_final[3*ii+2],times_unique[-1],0.1))[0]
            model_dec=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,np.arange(bestp_final[3*ii+2],times_unique[-1],0.1))[1]
            ax1.errorbar(hrs_to_mjd(Timea[ii]),np.sqrt((np.concatenate(Data2[2*ii]))**2+(np.concatenate(Data2[2*ii+1]))**2),\
                marker='o',color=col(coll(ii)),ls='',ms=10)
            ax1.plot(hrs_to_mjd(np.arange(bestp_final[3*ii+2],times_unique[-1],0.1)),np.sqrt(np.array(model_ra)**2+np.array(model_dec)**2),marker='',ls='-',color=col(coll(ii)))
ax1.set_xlabel('Time on 22/06/2015 (HH:MM)',fontsize=15)
ax1.set_ylabel('Angular Separation (mas)',fontsize=15)
ax1.tick_params(axis='both',which='minor',length=3,width=1)
ax1.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
locator = mdates.MinuteLocator(interval=15)
ax1.xaxis.set_major_locator(locator)
locator2 = mdates.MinuteLocator(interval=3)
ax1.xaxis.set_minor_locator(locator2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
minor_locator = AutoMinorLocator(4)
ax1.yaxis.set_minor_locator(minor_locator)
plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
ax1.tick_params(axis='both', which='major', labelsize=15,length=7,width=1)
ax1.tick_params(axis='both', which='minor', labelsize=15,length=3,width=1)
ax1.set_xlim(57195.43750000,57195.62500000)
ax1.set_ylim(-6,10)
plt.savefig(path_dir+'VLBA_positionfull.png',bbox_inches='tight')
print 'VLBA corrected ang sep vs time plot saved in '+path_dir

print 'Plotting position angles of components...'
fig=plt.figure(figsize=(15,10))
ax1=fig.add_subplot(1,1,1)
cm = plt.cm.get_cmap('jet',500)
coll=matplotlib.colors.Normalize(vmin=0,vmax=len(Timea))
col=matplotlib.cm.get_cmap('jet',500)
times_unique0=np.arange(1,28,0.1)
for ii in range(0,len(Timea)):
    if ii !=0:
        if bestp_final[3*ii+1]<0:#south comp
            model_ra=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,times_unique0)[0]
            model_dec=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,times_unique0)[1]
            plt.plot(model_ra[np.where(model_dec<0.)[0]],model_dec[np.where(model_dec<0.)[0]],color=col(coll(ii)),ls='-',lw=2)
        else:#north comp
            model_ra=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,times_unique0)[0]
            model_dec=bulk_motion_model([bestp_final[3*ii],bestp_final[3*ii+1],bestp_final[3*ii+2]]+jitt_best,times_unique0)[1]
            plt.plot(model_ra[np.where(model_dec>0.)[0]],model_dec[np.where(model_dec>0.)[0]],color=col(coll(ii)),lw=2)
plt.xlim(-1.5,1.5)
plt.errorbar(0,0,markersize=8,color='k',marker='o')
plt.xlabel('RA (mas)',fontsize=15)
plt.ylabel('DEC (mas)',fontsize=15)
plt.tick_params(axis='both',which='minor',length=3,width=1)
plt.tick_params(axis='both',which='major',labelsize=15,length=7,width=1)
plt.gca().invert_xaxis()
plt.ylim(-2,2)
plt.savefig(path_dir+'mcmc_PA.png',bbox_inches='tight')
print 'VLBA PA plot saved in '+path_dir

print ''
print '*********************************************'
print 'End of Script. Please inspect data products'
print '*********************************************'
