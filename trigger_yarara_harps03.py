#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:49:52 2020

@author: Cretignier Michael 
@university University of Geneva
"""

import sys
import os 
import matplotlib.pylab as plt
import numpy as np
import my_functions as myf
import my_classes as myc
import my_rassine_tools as myr
import getopt
import glob as glob
import time
import pandas as pd 
import astropy.coordinates as astrocoord
import pickle
from astropy import units as u

# =============================================================================
# TABLE
# =============================================================================
#
# stage = -2 (preprocessing importation from dace)
# stage = -1 (launch trigger RASSINE)
# stage = 0  (postprocessing)
# stage = 1 (basic operation and statistics)
# stage = 2 (cosmics correction)
# stage = 3 (fourier correction)
# stage = 4 (telluric water correction)
# stage = 5 (telluric oxygen correction)
# stage = 6 (telluric pca correction)
# stage = 7 (ccf moment/activity correction)
# stage = 8 (frog correction ghost A)
# stage = 9 (correction stitching)
# stage = 10 (frog correction ghost B)
# stage = 11 (TH contam)
# stage = 12 (correct continuum)
# stage = 13 (correct mad)
# stage = 14 (atmospheric parameters)
# stage = 15 (correct brute)
# stage = 16 (run kitcat)
# stage = 17 (atmospheric parameters 2)
# stage = 18 (activity proxy)
# stage = 19 (correction profile)
# stage = 20 (compute all ccf)
# stage = 21 (all map)
# stage = 22 (compute lbl)
# stage = 23 (compute dbd)
# stage = 24 (compute aba)
# stage = 25 (compute bt)
# stage = 26 (compute wbw)
# stage = 27 (1 year line rejection)
# stage = 28 (final graphic summary)
# stage = 29 (YARARA RV corrections proxies Ca,SAS1Y,WB)
# stage = 30 (YARARA RV corrections proxies PCA)
# stage = 31 (l1 periodogram)
# stage = 32 (hierarchical periodogram)
# stage = 33 (yarara qc file creation)
#
#
# =============================================================================


# =============================================================================
# PARAMETERS
# =============================================================================

star = 'HD21693'
ins = 'HARPS03'
input_product = 's1d'

stage = 1000          #begin at 1 when data are already processed 
stage_break = 28    #break included
cascade = True
close_figure = True
planet_activated = False
rassine_full_auto = 0
bin_length = 1
fast = False
reference = None
verbose=2
prefit_planet = False
drs_version = 'old'
sub_dico_to_analyse = 'matching_diff'
m_clipping = 3


if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:i:b:e:c:p:a:l:r:f:d:v:k:D:S:m:')
    for j in optlist:
        if j[0] == '-s':
            star = j[1]
        elif j[0] == '-i':
            ins = j[1]
        elif j[0] == '-b':
            stage = int(j[1])
        elif j[0] == '-e':
            stage_break = int(j[1])
        elif j[0] == '-c':
            cascade = int(j[1])      
        elif j[0] == '-p':
            planet_activated = int(j[1])  
        elif j[0] == '-a':
            rassine_full_auto = int(j[1]) 
        elif j[0] == '-l':
            bin_length = int(j[1])
        elif j[0] == '-r':
            reference = j[1]        
        elif j[0] == '-f':
            fast = int(j[1]) 
        elif j[0] == '-d':
            close_figure = int(j[1]) 
        elif j[0] == '-v':
            verbose = int(j[1])         
        elif j[0] == '-k':
            prefit_planet = int(j[1]) 
        elif j[0] == '-D':
            drs_version = j[1] 
        elif j[0] == '-S':
            sub_dico_to_analyse = j[1]         
        elif j[0] == '-m':
            m_clipping = int(j[1])
            
cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])
directory_yarara = root+'/Yarara/'    
directory_to_dace = directory_yarara + star + '/data/'+input_product+'/spectroDownload'
directory_rassine = '/'.join(directory_to_dace.split('/')[0:-1])+'/'+ins
directory_reduced_rassine = directory_rassine + '/STACKED/'
directory_workspace = directory_rassine + '/WORKSPACE/'

# =============================================================================
# BEGINNING OF THE TRIGGER
# =============================================================================

myr.print_iter(verbose)

obs_loc = astrocoord.EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) #HARPN

begin = time.time()
all_time= [begin]

time_step = {'begin':0}
button=0
instrument = ins[0:5]

def get_time_step(step):
    now = time.time()
    time_step[step] = now - all_time[-1]
    all_time.append(now)

def break_func(stage):
    button=1
    if stage>=stage_break:
        return 99
    else:
        if cascade:
            return stage + 1
        else:
            return stage
        
if stage==-2:
    
    # =============================================================================
    # EXTRACT INFORMATION IF IMPORTED FROM DACE
    # =============================================================================
    
    myr.move_extract_dace(directory_to_dace, instrument=instrument)
    
    if not os.path.exists('/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument):
        os.system('mkdir '+'/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument)
    
    myr.extract_table_dace(star, instrument=[instrument], update_table=False, auto_trend=True, m=m_clipping, degree_max=None, prefit_planet=prefit_planet, drs_version=drs_version)
    if os.path.exists('/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument+'/DACE_TABLE/Dace_extracted_table.csv'):
        if not os.path.exists(pd.read_csv('/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument+'/DACE_TABLE/Dace_extracted_table.csv',index_col=0)['fileroot'][0]):
            error = sys.exit()
    else:
        error = sys.exit()
    
    myr.split_instrument('/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument+'/', instrument=instrument)
    
    if not len(glob.glob('/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument+'/*.fits')):
        os.system('rm -rf '+'/'.join(directory_to_dace.split('/')[0:-1])+'/'+instrument+'/'+' 2>/dev/null')

    plt.close('all')
    get_time_step('dace_extraction')
    stage = break_func(stage)

if stage==-1:
    
    # =============================================================================
    # RUN RASSINE TRIGGER
    # =============================================================================
    
    #in topython terminal, change the Trigger file 
    print(' python Rassine_trigger.py -s %s -i %s -a %s -b %s -d %s -l 0.01 -o %s'%(star, [instrument,'ESPRESSO'][drs_version!='old'], str(rassine_full_auto), str(bin_length), ins, directory_rassine+'/'))
    os.system('python Rassine_trigger.py -s %s -i %s -a %s -b %s -d %s -l 0.01 -o %s'%(star, [instrument,'ESPRESSO'][drs_version!='old'], str(rassine_full_auto), str(bin_length), ins, directory_rassine+'/'))
    get_time_step('rassine')
    
    reduced_files = glob.glob(directory_reduced_rassine+'RASSINE*.p')

    if len(reduced_files):   
        if not os.path.exists(directory_workspace):
            os.system('mkdir '+directory_workspace)
            
        for k in reduced_files:
            os.system('cp '+k+' '+directory_workspace)              

    stage = break_func(stage)


# =============================================================================
# RUN YARARA
# =============================================================================

try:
    sts = myr.spec_time_series(directory_workspace)
    #file_test = sts.import_spectrum()
    sts.planet = planet_activated
    sts.instrument = ins.split('_planet')[0]

    print('------------------------\n STAR LOADED : %s \n------------------------'%(sts.starname))
    
    if fast:
        print('\n [INFO] Fast analysis %s %s reference spectrum launched...\n'%(ins,['with','without'][int(reference is None)]))
    else:
        print('\n [INFO] Complete analysis %s %s reference spectrum launched...\n'%(ins,['with','without'][int(reference is None)]))
    
    
except:
    if stage>=0:
        print('\n [ERROR] No files found')
        stage = 9999

if stage==999:
    sts.yarara_recreate_dico('matching_fourier')
    sts.yarara_recreate_dico('matching_telluric')
    sts.yarara_recreate_dico('matching_oxygen')
    sts.yarara_recreate_dico('matching_stitching')    
    sts.yarara_recreate_dico('matching_ghost_a')
    sts.yarara_recreate_dico('matching_ghost_b')    
    sts.yarara_recreate_dico('matching_thar')    
    sts.yarara_recreate_dico('matching_smooth')    
    
if stage==777: #to rerun the pipeline automatically from the last step in case of a crash (however pickle may be corrupted...)
    sts.import_dico_tree()
    sts.import_material()
    last_dico = np.array(sts.dico_tree['dico'])[-1]
    if 'reference_spectrum_backup' in sts.material.keys():
        reference='master'
    else:
        reference=None
        stage_break = 14
    vec = {'matching_diff':1,'matching_cosmics':2,'matching_fourier':3,'matching_telluric':4,'matching_oxygen':5,'matching_pca':6,
           'matching_activity':7,'matching_ghost_a':8,'matching_stitching':9,'matching_ghost_b':10,'matching_thar':11,'matching_berv':11,
           'matching_smooth':12,'matching_mad':13,'matching_brute':14,'matching_morpho':25,'matching_shell':27,'matching_color':27,'matching_empca':28}
    stage = vec[last_dico]+1
    
        
# =============================================================================
# LOAD DATA IN YARARA
# =============================================================================

if stage==0:
    #sts.yarara_inject_planet()
    sts.yarara_simbad_query()

    bin_length = sts.yarara_get_bin_length()

    sts.import_dace_table_rv(bin_length=bin_length,dbin=0)
    sts.import_table()
    
    snr = myc.tableXY(sts.table.jdb,sts.table.snr)
    plt.figure() ; snr.myscatter() ; plt.axhline(y=np.nanpercentile(snr.y,25),color='k') ; plt.axhline(y=np.nanpercentile(snr.y,50),color='k') ; plt.axhline(y=np.nanpercentile(snr.y,75),color='k')

    sts.flux_error(ron=11)
    sts.continuum_error()
    
    sts.yarara_check_rv_sys()
    sts.yarara_check_fwhm()
    plt.close('all')
    
    sts.yarara_ccf(mask=sts.mask_harps, ccf_oversampling=1, plot=True, save=True, rv_range=None) 
    
    sts.yarara_correct_secular_acc(update_rv=True)    

    lim_inf = np.nanpercentile(snr.y,50) - 1.5*myf.IQ(snr.y)
    
    for lim in [100, 75, 50, 35, 20]:
        if (lim_inf < lim)&(np.nanpercentile(snr.y,16)>lim):
            lim_inf = lim
            break
        
    if lim_inf<0:
        lim_inf = 35
    
    print('Spectra under SNR %.0f supressed'%(lim_inf))        
    sts.supress_low_snr_spectra(snr_cutoff=lim_inf, supress=False)
    
    sts.yarara_supress_doubtful_spectra(supress=False)
    
    sts.supress_time_spectra(num_min=None, num_max=None)
    
    #sts.split_instrument(instrument=ins)
            
    if close_figure:
        plt.close('all')
    
    get_time_step('preprocessing')        
    stage = break_func(stage)
        
if stage==1:
    #STATISTICS
    sts.yarara_simbad_query()
    #sts.yarara_star_info(sp_type='K1V', Mstar=1.0, Rstar=1.0)
    
    sts.yarara_add_step_dico('matching_diff',0) #sts.yarara_add_step_dico('matching_brute',0,chain=True)
    
    #COLOR TEMPLATE
    print('\n Compute bolometric constant')
    sts.yarara_flux_constant()
    sts.yarara_color_template()
    
    sts.yarara_map_1d_to_2d(instrument = ins)
    
    #ERROR BARS ON FLUX AND CONTINUUM
    print('\n Add flux errors')
    sts.yarara_non_zero_flux()
    sts.flux_error(ron=11)
    sts.continuum_error()

    #BERV SUMMARY
    sts.yarara_berv_summary(sub_dico='matching_diff', continuum='linear', dbin_berv = 3, nb_plot=2)

    #ACTIVITY
    print('\n Compute activity proxy')
    sts.yarara_activity_index(sub_dico='matching_diff', continuum = 'linear')

    #table
    sts.yarara_obs_info(kw=['instrument',[ins]])
    sts.import_table()
    
    print('\n Make SNR statistics figure')    
    sts.snr_statistic()
    print('\n Make DRS RV summary figure')
    sts.dace_statistic()

    sts.yarara_transit_def(period=100000,T0=0,duration=0.0001)

    #CROPING 
    if False:
        sts.yarara_time_variations(sub_dico='matching_diff',
                                   wave_min=3700,
                                   wave_max=4000)
        
        sts.yarara_time_variations(sub_dico='matching_diff',
                                   wave_min=6800,
                                   wave_max=7000)
    
    print('\n Crop spectra')    
    sts.yarara_cut_spectrum(wave_min=3805.00,wave_max=6865.00)
    
    if close_figure:
        plt.close('all')

    bin_length = sts.yarara_get_bin_length()
    
    sts.import_drift_night(ins,bin_length=bin_length, drs_version=drs_version)

    sts.import_rv_dace(ins,calib_std=0.7,bin_length=bin_length, drs_version=drs_version)

    sts.yarara_plot_rcorr_dace(bin_length=bin_length, detrend=2)
    
    get_time_step('statistics')        

    stage = break_func(stage)
     
# =============================================================================
# COSMICS
# =============================================================================

if stage==2:
    #CORRECT COSMICS
    sts.yarara_correct_cosmics(sub_dico = 'matching_diff', 
                               continuum = 'linear', 
                               k_sigma = 5)
    
    if reference is None:        
        #MEDIAN MASTER
        sts.yarara_median_master(sub_dico = 'matching_cosmics', 
                         continuum = 'linear',
                         method = 'mean', #if smt else than max, the classical weighted average of v1.0
                         bin_berv = 10, 
                         bin_snr = 1000,
                         smooth_box=1,
                         jdb_range=[54194,55037,0])
        
        sts.yarara_telluric(sub_dico = 'matching_cosmics', 
                            continuum = 'linear',
                            reference='norm',
                            ratio=True)
        
        sts.yarara_master_ccf(sub_dico='matching_cosmics',name_ext='_telluric')

        sts.yarara_median_master(sub_dico = 'matching_cosmics', 
                                 continuum = 'linear',
                                 method = 'mean', #if smt else than max, the classical weighted average of v1.0
                                 bin_berv = 10, 
                                 bin_snr = 1000,
                                 smooth_box=1,
                                 jdb_range=[54194,55037,0])

    if close_figure:
        plt.close('all')

    get_time_step('matching_cosmics')            
    
    stage = break_func(stage)
    
# =============================================================================
# FOURIER
# =============================================================================
    
if stage==3:
    #CORRECT INTERFERENCE PATTERN

    if reference is None:        
        #MEDIAN MASTER
        sts.yarara_telluric(sub_dico = 'matching_cosmics', 
                    continuum = 'linear',
                    reference='norm',
                    ratio=True,
                    normalisation='left')   

        
        sts.yarara_median_master(sub_dico = 'matching_cosmics', 
                                 continuum = 'linear',
                                 method = 'mean', #if smt else than max, the classical weighted average of v1.0
                                 bin_berv = 10, 
                                 bin_snr = 1000,
                                 smooth_box=1,
                                 jdb_range=[54194,55037,0])
    
    
    #CORRECT PATTERN IN FOURIER SPACE
    sts.yarara_correct_pattern(sub_dico = 'matching_cosmics', 
                               continuum = 'linear', 
                               reference = 'master', 
                               correct_blue = True, 
                               correct_red = True,
                               jdb_range=[54194,55037],
                               width_range=[1.3,1.7])
    
    if close_figure:
        plt.close('all')

    get_time_step('matching_fourier')            
                    
    stage = break_func(stage)

# =============================================================================
# TELLURIC WATER
# =============================================================================

if stage==4:
    #CORRECT TELLURIC WATER
    file_test = sts.import_spectrum()
    
    if 'matching_fourier' in file_test.keys():
        sb = 'matching_fourier'
    else:
        sb = 'matching_cosmics'        

    #TELLURIC FOR MEDIAN MASTER

    if reference is None:
        #MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(sub_dico = sb, 
                                 continuum = 'linear',
                                 method = 'mean', #if smt else than max, the classical weighted average of v1.0
                                 bin_berv = 10, 
                                 bin_snr = 1000,
                                 smooth_box=1)

    sts.yarara_telluric(sub_dico = sb, 
                        mask='telluric',
                        continuum = 'linear',
                        reference='norm',
                        ratio=True,
                        normalisation='left')   


    if reference is None:
        #MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(sub_dico = sb, 
                                 continuum = 'linear',
                                 method = 'mean', #if smt else than max, the classical weighted average of v1.0
                                 bin_berv = 10, 
                                 bin_snr = 1000,
                                 smooth_box=1)


    sts.yarara_telluric(sub_dico = sb, 
                        mask='h2o',
                        continuum = 'linear',
                        reference='norm',
                        ratio=True,
                        normalisation='left')   
    
    
    #CORRECT WATER TELLURIC WITH CCF
    sts.yarara_correct_telluric_proxy(sub_dico = sb,
                                      sub_dico_output='telluric',
                                      reference = 'master',
                                      proxies_corr = ['h2o_depth','h2o_fwhm'], 
                                      wave_min_correction = 4400,
                                      min_r_corr = 0.4,
                                      sigma_ext=2)

    if close_figure:
        plt.close('all')

    get_time_step('matching_telluric')            
        
    stage = break_func(stage)


# =============================================================================
# TELLURIC OXYGEN
# =============================================================================

if stage==5:
    #CORRECT OXYGEN WATER
    file_test = sts.import_spectrum()
    
    if 'matching_fourier' in file_test.keys():
        sb = 'matching_fourier'
    else:
        sb = 'matching_cosmics'  

    if reference is None:
        #MEDIAN MASTER FOR TELLURIC CORRECTION
        sts.yarara_median_master(sub_dico = sb, 
                                 continuum = 'linear',
                                 method = 'mean', #if smt else than max, the classical weighted average of v1.0
                                 bin_berv = 10, 
                                 bin_snr = 1000,
                                 smooth_box=1)

    #TELLURIC FOR MEDIAN MASTER
    sts.yarara_telluric(sub_dico = sb, 
                        mask='o2',
                        continuum = 'linear',
                        reference='norm',
                        ratio=True,
                        normalisation='left')   

    
    #CORRECT WATER TELLURIC WITH CCF
    sts.yarara_correct_telluric_proxy(sub_dico = 'matching_telluric',
                                      sub_dico_output='oxygen',
                                      reference = 'master',
                                      proxies_corr = ['h2o_depth','h2o_fwhm','o2_depth','o2_fwhm'], 
                                      wave_min_correction = 4400,
                                      min_r_corr = 0.4,
                                      sigma_ext=2)

    sts.yarara_correct_oxygen(sub_dico = 'matching_oxygen', 
                               oxygene_bands = [[5787,5835]], 
                               reference = 'master', 
                               continuum = 'linear')

    if close_figure:
        plt.close('all')

    get_time_step('matching_oxygen')            
        
    stage = break_func(stage)

# =============================================================================
# TELLURIC PCA
# =============================================================================

if stage==6:
    #CORRECT TELLURIC WITH PCA    
    file_test = sts.import_spectrum()

    if 'matching_fourier' in file_test.keys():
        dic_ref = 'matching_fourier'
    else:
        dic_ref = 'matching_cosmics' 
    
    if reference is None:
        sts.yarara_median_master(sub_dico = 'matching_oxygen', 
                                 continuum = 'linear',
                                 method = 'mean' #if smt else than max, the classical weighted average of v1.0
                                 )
    
    sts.yarara_correct_telluric_gradient(sub_dico_detection = dic_ref,
                                         sub_dico_correction = 'matching_oxygen', 
                                         inst_resolution = 110000, 
                                         continuum = 'linear',
                                         reference = 'master',
                                         equal_weight = True,
                                         wave_min_correction = 4400,
                                         calib_std = 1e-3,
                                         nb_pca_comp_kept = None,
                                         nb_pca_max_kept = 3)   
    
    if close_figure:
        plt.close('all')

    get_time_step('matching_pca')            
        
    stage = break_func(stage)

# =============================================================================
# CCF MOMENTS
# =============================================================================   

ref = ['master','median'][reference is None]

# =============================================================================
# CCF MOMENTS
# =============================================================================

if stage==7:
    #CORRECT ACTIVITY WITH PROXIES + CCF MOMENT
    print('\n Compute activity proxy')
    sts.yarara_activity_index(sub_dico='matching_pca', continuum = 'linear')

    sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_pca', ccf_oversampling=1, rv_range=None) 
    
    proxy = sts.yarara_determine_optimal_Sindex()
    print('\n Optimal proxy of activity : %s'%(proxy))
    
    sts.yarara_correct_activity(sub_dico = 'matching_pca', 
                                reference = ref,
                                proxy_corr = [proxy,'ccf_fwhm','ccf_contrast'], 
                                smooth_corr=1)

    if close_figure:
        plt.close('all')

    get_time_step('matching_activity')            
    
    stage = break_func(stage)
   
# =============================================================================
# GHOST
# =============================================================================
     
if stage==8:
    #CORRECT FROG
    sts.yarara_correct_borders_pxl(pixels_to_reject=np.hstack([np.arange(1,6), np.arange(4092,4097)]))
    sts.yarara_produce_mask_frog(frog_file=root+'/Python/Material/Ghost_'+ins.split('_')[0]+'.p')
    #sts.yarara_produce_mask_stitching(frog_file=root+'/Python/Material/Ghost_'+ins+'.p')
            
    sts.yarara_correct_frog(correction = 'ghost_a',
                            sub_dico = 'matching_activity', 
                            reference = ref,
                            berv_shift='berv', treshold_contam=3.5, equal_weight=False, 
                            pca_comp_kept=3, complete_analysis=False, algo_pca='pca')
    
    if close_figure:
        plt.close('all')

    get_time_step('matching_ghost_a')            
                
    stage = break_func(stage)

# =============================================================================
# STITCHIHG
# =============================================================================

if stage==9:
    #CORRECT FROG
    sts.yarara_correct_stitching(sub_dico='matching_ghost_a', smooth_box=3, reference=ref)

    if close_figure:
        plt.close('all')

    get_time_step('matching_stitching')            
                
    stage = break_func(stage)
        
# =============================================================================
# GHOSTB
# =============================================================================
     
if stage==10:
    #CORRECT GHOST B CONTAM
    sts.yarara_correct_frog(correction = 'ghost_b',
                            sub_dico = 'matching_stitching', 
                            reference = ref,
                            berv_shift='berv', wave_max_train=4100, equal_weight=True,
                            pca_comp_kept=2, treshold_contam=2.5, complete_analysis=False, algo_pca='pca')
    
    if close_figure:
        plt.close('all')

    get_time_step('matching_ghost_b')            

    stage = break_func(stage)

# =============================================================================
# THAR CONTAM
# =============================================================================
        
if stage==11:
    #CORRECT TH CONTAM    
    sts.yarara_correct_frog(correction = 'thar',
                            sub_dico = 'matching_ghost_b', 
                            reference = ref,
                            berv_shift='berv', wave_max_train=4100, equal_weight=True, 
                            pca_comp_kept=2, treshold_contam=1, complete_analysis=False, rcorr_min=0.4)

    if close_figure:
        plt.close('all')
        
    if False:
        sts.yarara_correct_berv_line(sub_dico = 'matching_thar',
                                     continuum = 'linear',
                                     reference = ref,
                                     jdb_range=[[0,100000,2,4],[54988,55021,0,1]], 
                                     method='weighted',wave_min=4170,wave_max=4230)
    
#    sts.yarara_correct_contam_TH(sub_dico = None, 
#                             continuum = 'linear',
#                             nb_pca_comp=25,
#                             equal_weight=True,
#                             nb_pca_max_kept = 3,
#                             wave_max_train = 4500,
#                             detection='outliers') #or frog
    

    if close_figure:
        plt.close('all')

        
    get_time_step('matching_thar')            

    stage = break_func(stage)
          
# =============================================================================
# CONTINUUM
# =============================================================================
      
if stage==12:
    #CORRECT CONTINUUM 
    file_test = sts.import_spectrum()

    if 'matching_berv' in file_test.keys():
        sub_dico = 'matching_berv'
    else:
        sub_dico = 'matching_thar' 
    
    sts.yarara_correct_smooth(sub_dico=sub_dico,
                           continuum='linear',
                           reference=ref,
                           window_ang = 5)
  
    if reference is None:
        sts.yarara_retropropagation_correction(correction_map='matching_smooth',
                                    sub_dico='matching_cosmics',
                                    continuum='linear')

    if close_figure:
        plt.close('all')

    get_time_step('matching_smooth')            
        
    stage = break_func(stage)    
    
# =============================================================================
# MAD
# =============================================================================
      
if stage==13:
    #CORRECT MAD
    sts.yarara_correct_mad(sub_dico='matching_smooth',
                           continuum='linear',
                           k_sigma = 2,
                           k_mad = 2,
                           n_iter = 1,
                           ext=['0','1'][int(reference=='master')])

    spectrum_removed = (sts.counter_mad_removed>[0.15,0.15][int(reference=='master')])
    sts.supress_time_spectra(liste=spectrum_removed)

    sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_mad', ccf_oversampling=1, rv_range=None) 
            
    if close_figure:
        plt.close('all')

    get_time_step('matching_mad')            
        
    stage = break_func(stage)
   
    
# =============================================================================
# STELLAR ATMOS
# =============================================================================
      
if stage==14:
    #TEMPERATURE + MODEL FIT
        
    sts.yarara_temperature_variation(plot = True, sub_dico = 'matching_pca')
    
    sts.yarara_median_master(sub_dico = 'matching_mad', 
                             continuum = 'linear',
                             method = 'median', #if smt else than max, the classical weighted average of v1.0
                             supress_telluric=False,
                             shift_spectrum=False)
    
    if reference!='master':
        sts.import_material()
        load = sts.material
        load['reference_spectrum_backup'] = load['reference_spectrum']
        load.to_pickle(sts.directory+'Analyse_material.p')
    
    sts.yarara_cut_spectrum(wave_min=None, wave_max=6834)
    
    sts.yarara_stellar_atmos(sub_dico='matching_diff', reference='master', continuum='linear') #sub_dico only for errorbars
    
    sts.yarara_correct_continuum_absorption(model=None, T=None, g=None)

    sts.yarara_snr_curve()
    sts.snr_statistic(version=2)
    sts.yarara_kernel_caii(contam=True, mask='CaII_HD128621', power_snr=None, noise_kernel='unique', wave_max=None, doppler_free=False)
        
    if close_figure:
        plt.close('all')
        
    get_time_step('stellar_atmos1')            
    
    stage = break_func(stage)   
    
# =============================================================================
# BRUTE
# =============================================================================
    
if stage==15:
    #CORRECT BRUTE    
    sts.yarara_correct_brute(sub_dico='matching_mad',
                             continuum='linear',
                             min_length=3,
                             k_sigma=2,
                             percent_removed=10, #None if sphinx 
                             ghost2=False,
                             borders_pxl=True)

    if close_figure:
        plt.close('all')

    get_time_step('matching_brute')            
        
    stage = break_func(stage)

# =============================================================================
# KITCAT
# =============================================================================

if stage==16:
    #KITCAT MASK

    if os.path.exists(sts.dir_root+'CCF_MASK/CCF_kitcat_mask_'+sts.starname+'.fits'):
        os.system('rm '+sts.dir_root+'CCF_MASK/CCF_kitcat_mask_'+sts.starname+'.fits')
    if os.path.exists(sts.dir_root+'CCF_MASK/CCF_kitcat_cleaned_mask_'+sts.starname+'.fits'):
        os.system('rm '+sts.dir_root+'CCF_MASK/CCF_kitcat_cleaned_mask_'+sts.starname+'.fits')
    
    sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_mad', ccf_oversampling=1, rv_range=None) 
    
    sts.import_telluric(ext='')
    
    command_line = sts.yarara_preprocess_mask(sub_dico_ref='matching_diff', sub_dico='matching_mad', method='stack', 
                                              plot=True, template_telluric=False, shift_spectrum=True)
    
    myf.make_sound('Kitcat will be launched in a few seconds')
    #take the line code printed by yarara
    os.system(command_line)

    sts.import_kitcat() ; cut = np.min(sts.kitcat['catalogue']['wave']) + 0.5*(np.max(sts.kitcat['catalogue']['wave']) - np.min(sts.kitcat['catalogue']['wave']))
    sts.yarara_detector(wave_cut=[cut])

    sts.kitcat_match_spectrum(windows='large',clean=False)
    
    sts.yarara_ccf(mask = 'kitcat_mask_'+sts.starname+'.p', plot=True, sub_dico='matching_mad', ccf_oversampling=1, rv_range = None) 
    sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_mad', ccf_oversampling=1, rv_range = None) 
    
    sts.yarara_kitcat_plot(wave_min=4435,wave_max=4445, nb_bins=30, sub_dico='matching_mad')
    
    sts.kitcat_statistic_telluric(clean=False,telluric_ext='')
    
    if False:
        sts.kitcat_produce_generic_mask(telluric_tresh=0.01,weight_col='weight_rv',clean=False,rv_sys=None)    
    
    if close_figure:
        plt.close('all')
    
    get_time_step('kitcat')            
    
    stage = break_func(stage)
        
# =============================================================================
# STELLAR ATMOS2
# =============================================================================
    
if stage==17:
    #COMPUTE MORE ATMOS PARAM
    ccf_output = sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_mad', save=False, rv_range = None, delta_window=5, ccf_oversampling=3)     
    sts.yarara_master_ccf(sub_dico='matching_mad',name_ext='_mask_HARPS')
    
    ccf_output = sts.yarara_ccf(mask = 'kitcat_cleaned_mask_'+sts.starname+'.p', plot=True, sub_dico='matching_mad', save=False, rv_range = None, delta_window=5, ccf_oversampling=3)
    sts.yarara_master_ccf(sub_dico='matching_mad')
    
    sts.yarara_vsin(fwhm_star=sts.fwhm, fwhm_inst=3.5, vmicro=None)
    sts.yarara_metallicity(continuum_tresh=0.95)
    
    if close_figure:
        plt.close('all')

    get_time_step('stellar_atmos2') 

    stage = break_func(stage)

# =============================================================================
# ACTIVITY PROXY
# =============================================================================

if stage==18:
    #WB ACTIVITY
    sts.yarara_activity_index(sub_dico='matching_pca', continuum = 'linear', substract_map=['ghost_a','ghost_b','thar','mad','stitching','smooth'])
    
    sts.yarara_wb_ca2(sub_dico = 'matching_pca', substract_proxy='CaII', optimize='both', window_extraction = None, window_core = None,  substract_map=['ghost_a','ghost_b','thar','stitching','smooth'])
    
    # sts.yarara_wb_h1(sub_dico = 'matching_activity', window_extraction=[0.7, 0.25, 0.15],window_core=[None, None, None],substract_map=['ghost_a','ghost_b','thar','stitching'])
    # sas1year, sas1year_smooth = sts.lbl_create_proxy(criterion=['wave',3], sub_dico='matching_mad', window_bin=0)
    
    ccf_output = sts.yarara_ccf(mask = 'kitcat_cleaned_mask_'+sts.starname+'.p', plot=True, sub_dico='matching_mad', save=False, ccf_oversampling=3) 
    bis_pca = sts.yarara_ccf_bis(ccf_output,sub_dico='matching_mad',reduction='pca')
    
    bis = bis_pca[0] #ccf_output['vspan']
    bis2 = bis_pca[1]
    sts.yarara_obs_info(kw=['BIS',bis.y])
    sts.yarara_obs_info(kw=['BIS_std',bis.yerr])
    sts.yarara_obs_info(kw=['BIS2',bis2.y])
    sts.yarara_obs_info(kw=['BIS2_std',bis2.yerr])
    
    ccf_output['rv'].yerr+=0.7
    
    sts.yarara_kde_mask()
    sts.yarara_cb(sub_dico_rv='matching_mad', vec_corr=ccf_output['rv'], sig1=1.5, sig2=2.5)
    #sts.yarara_cb2(sub_dico_rv='matching_mad')

    sts.yarara_kernel_caii(contam=True, mask='CaII_HD128621', power_snr=None, noise_kernel='unique', wave_max=None, doppler_free=False)
    sts.yarara_kernel_wb(contam=False, mask='WB_HD128621', power_snr=None, noise_kernel='unique', wave_max=None, doppler_free=True)

    if close_figure:
        plt.close('all')  

    get_time_step('activity proxy')                   

    stage = break_func(stage)
    
# =============================================================================
# ALL CCF
# =============================================================================

if stage==19:
    #COMPUTE ALL DICTIONNARY CCF    
    for mask_ccf in [sts.mask_harps,'kitcat_mask_'+sts.starname+'.p']:
        sts.yarara_ccf_all(continuum='linear',
                           mask = mask_ccf,
                           rv_range = None,
                           ccf_oversampling=1,
                           dicos = ['all','light'][int(fast)])
    
    sts.yarara_ccf_replace_nan()
        
    if False:
        for sub_dico in ['matching_smooth','matching_mad']:
            sts.yarara_ccf(mask = 'kitcat_mask_'+sts.starname+'.p', plot=True, sub_dico=sub_dico, ccf_oversampling=1, rv_range = None) 
            sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico=sub_dico, ccf_oversampling=1, rv_range = None) 
            
    get_time_step('all_ccf')
    
    stage = break_func(stage)

# =============================================================================
# ALL MAP
# =============================================================================       

if stage==20:
    print('\n')
    sts.yarara_map_all(wave_min=3920, wave_max=3980)
    plt.close('all') 
    
    sts.yarara_map_all(wave_min=5730, wave_max=5800)
    plt.close('all') 

    sts.yarara_comp_all(analysis='h2o_1')
    sts.yarara_comp_all(analysis='h2o_2')
    sts.yarara_comp_all(analysis='o2_1')
    sts.yarara_comp_all(analysis='ha')

    if close_figure:
        plt.close('all')  

    get_time_step('all_map')                   

    stage = break_func(stage)
        
# =============================================================================
# LBL
# =============================================================================
    
if stage==21:
    #COMPUTE LBL
    sts.yarara_lbl(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3)
    
    sts.yarara_lbl_iter(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, flux_max=0.95, oversampling=10, normalised=False, rm_outliers=False)
 
    sts.instrument = ins
    sts.kitcat_statistic_lbl(clean=False,telluric_tresh=1)      
 
    get_time_step('lbl')                   
    
    stage = break_func(stage)
              
    
if fast:
    dicos = ['matching_mad']
else:
    dicos = ['matching_mad','matching_activity','matching_pca','matching_diff']

# =============================================================================
# DBD
# =============================================================================

if stage==22:    
    #COMPUTE DBD
    sts.yarara_dbd(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, all_dico=dicos)
    
    get_time_step('dbd') 
    
    stage = break_func(stage)

# =============================================================================
# ABA
# =============================================================================
    
if stage==23:
    #COMPUTE ABA
    sts.yarara_aba(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, all_dico=dicos)
    
    get_time_step('aba') 
    
    stage = break_func(stage)

# =============================================================================
# BT
# =============================================================================
    
if stage==24:
    #COMPUTE BT
    if False: #no more used since useless and crash on lesta because of memory issue (division by 0 ?)
        sts.yarara_bt(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, all_dico=dicos)
    
    get_time_step('bt') 
    
    stage = break_func(stage)

# =============================================================================
# WBW
# =============================================================================
     
if stage==25:
    #COMPUTE WBW
    sts.yarara_wbw(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std = 1e-3, treshold_sym=0.40, treshold_asym=0.40, all_dico=dicos)

    if close_figure:
        plt.close('all')
    
    get_time_step('wbw') 
    
    stage = break_func(stage)

# =============================================================================================================================
# End of the flux corrections from YARARA V1 | YARARA V2 is beginning RV corrections
# =============================================================================================================================

# =============================================================================
# CORRECTION LINE MORPHO + SUPRESSION 1Y LINE
# =============================================================================

if stage==26:
    sts.yarara_lbl(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, all_dico=['matching_mad'])

    sts.yarara_lbl_iter(kitcat=sts.dir_root+'KITCAT/kitcat_mask_'+sts.starname+'.p', calib_std=1e-3, flux_max=0.95, oversampling=10, all_dico=['matching_mad'], normalised=False, rm_outliers=False)
    
    sts.lbl_fit_base('matching_morpho', sub_dico='matching_mad', kw_dico='lbl', proxies=['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=0)
    sts.lbl_fit_base('matching_morpho', sub_dico='matching_mad', kw_dico='lbl_iter', proxies=['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=0)
 
    sts.lbl_supress_1year(sub_dico='matching_morpho', kw_dico='lbl', fap=1, p_min=0, add_step=0)
    sts.lbl_supress_1year(sub_dico='matching_morpho', kw_dico='lbl_iter', fap=1, p_min=0, add_step=0)
    
    sts.lbl_polar_1year(kw_dico='lbl',sub_dico='matching_morpho')
    sts.lbl_polar_1year(kw_dico='lbl_iter',sub_dico='matching_morpho')

    sts.yarara_sas(sub_dico='matching_morpho', kw_dico='lbl_iter', g1=[['line_depth','<',0.5]],g2=[['line_depth','>',0.5]],Plot=False)
    
    get_time_step('matching_morpho')                   

    if close_figure:
        plt.close('all')
    
    stage = break_func(stage)  


# =============================================================================
# FINAL PLOT
# =============================================================================       

if stage==27:
    sts.yarara_curve_rv(poly_deg=0, metric='gaussian',last_dico='matching_1y')
    
    if close_figure:
        plt.close('all')
    
    sts.yarara_plot_all_rv('CCF_kitcat_mask_'+sts.starname, 'matching_morpho', ofac=10, photon_noise=0.7, deg_detrending=2, p_min=0, method='iterative')

    if close_figure:
        plt.close('all')
    
    get_time_step('final_plot_v1')
    
    stage = break_func(stage)

        
# =============================================================================
# POST RV CORRECTION (CA,WB,BIS)
# =============================================================================       

if stage==28:
        
    # sts.yarara_ccf_fit('matching_ca', sub_dico = 'matching_mad', proxies = ['CaII'], time_detrending = 0)    
    # sts.lbl_fit_base('matching_ca', sub_dico='matching_mad', kw_dico='lbl', proxies=['CaII','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=4)
    # sts.lbl_fit_base('matching_ca', sub_dico='matching_mad', kw_dico='lbl_iter', proxies=['CaII','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=4)
    
    # sts.yarara_ccf_fit('matching_wb', sub_dico = 'matching_mad', proxies = ['CaII','WB'], time_detrending = 0)
    # sts.lbl_fit_base('matching_wb', sub_dico='matching_mad', kw_dico='lbl', proxies=['CaII','WB','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=5)
    # sts.lbl_fit_base('matching_wb', sub_dico='matching_mad', kw_dico='lbl_iter', proxies=['CaII','WB','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=5)
    
    # sts.yarara_ccf_fit('matching_cb', sub_dico = 'matching_mad', proxies = ['CaII','CB'], time_detrending = 0)
    # sts.lbl_fit_base('matching_cb', sub_dico='matching_mad', kw_dico='lbl', proxies=['CaII','CB','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=6)
    # sts.lbl_fit_base('matching_cb', sub_dico='matching_mad', kw_dico='lbl_iter', proxies=['CaII','CB','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=6)
    
    # sas1year, sas1year_smooth = sts.lbl_create_proxy(criterion=['period',365.25,'K'], group='anticorr', sub_dico='matching_wb', window_bin=20,season=0,col=2) #new recipes less absorbing 120 days planets
    # sts.yarara_obs_info(kw=['SAS1Y',sas1year.y])
    # sts.yarara_obs_info(kw=['SAS1Y_std',sas1year.yerr])
    
    # sts.yarara_ccf_fit('matching_bis', sub_dico = 'matching_brute', proxies = ['CaII','WB','BIS'], time_detrending = 0)
    # sts.lbl_fit_base('matching_bis', sub_dico='matching_mad', proxies=['CaII','WB','BIS','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=4)
    
    # sts.yarara_ccf_fit('matching_1y', sub_dico = 'matching_brute', proxies = ['CaII','WB','BIS','SAS1Y'], time_detrending = 0)
    # sts.lbl_fit_vec('matching_bis', sub_dico = 'matching_brute', base_vec = ['CaII','WB','BIS'], time_detrending = 0, add_step=2)
    # sts.lbl_fit_base('matching_1y', sub_dico='matching_mad', proxies=['CaII','WB','BIS','SAS1Y','dbd_0','aba_6','aba_7'], time_detrending=0, add_step=5)
    # sts.lbl_fit_sinus(365.25, sub_dico='matching_morpho', season=0, good_morpho=False, kde=False, kw_dico = 'lbl', col = 0, color_axis='line_depth')
    # sts.lbl_polar_1year() 
          
    print('\n Determine rotationnal period')
    # vec_lbl = sts.import_ccf_timeseries('LBL_kitcat_mask_'+sts.starname,'matching_cb','rv')
    # sts.yarara_obs_info(kw=['ccf_rv',vec_lbl.y/1000])
    # sts.yarara_obs_info(kw=['ccf_rv_std',vec_lbl.yerr/1000])
    sts.yarara_stellar_rotation(windows=100, min_nb_points=25, prot_max=100, rhk=1, kernel_rhk=0, ca2=0, ha=1, h=0, wb=1, mg1=0, nad=0, bis=1, cb=1, rv=1)
    sts.yarara_star_info(Pmag=['YARARA',np.round(sts.Pmag,1)])

    sts.yarara_ccf(mask = 'kitcat_mask_'+sts.starname+'.p', plot=True, sub_dico='matching_mad', ccf_oversampling=1, rv_range = None)

    sts.yarara_median_master(sub_dico = 'matching_mad', 
                     continuum = 'linear',
                     method = 'median', #if smt else than max, the classical weighted average of v1.0
                     supress_telluric=False,
                     shift_spectrum=True,
                     mask_percentile = [None,50])

    #sts.yarara_curve_rv(poly_deg=2, metric='gaussian',last_dico='matching_cb')

    if close_figure:
        plt.close('all')

    get_time_step('matching_wb')                   

    stage = break_func(stage)  
    
       
if stage==29:

    kw_dico = 'lbl_iter'
    nb_comp_shell = 3
    nb_comp_color = 3
    nb_comp_pca = 3
    alpha_crit = 1 # 1% of significance alpha
    z_crit = 0.25
    nb_period = 1000
    treshold_percent = 95 # put higher than 100 to not use cross validation 
    DBD = True
    min_r_percentile = 25
    
    base_tot = []
    sts.simu = {}
    sub_dico1 = 'matching_mad'
    
    if nb_comp_shell:
                
        # =============================================================================
        # shell correction 
        # =============================================================================
        
        for comp_kept in [5]:    
            sts.yarara_correct_shell(sub_dico='matching_mad', 
                                 reduction='pca', 
                                 reference='master', 
                                 continuum='linear', 
                                 substract_map=[], 
                                 continuum_absorption=True, 
                                 wave_min=None, 
                                 wave_max=None, 
                                 m=2, m_shell=3, kind='inter', 
                                 nb_comp=10,
                                 nb_comp_kept=comp_kept,
                                 min_element_nb=30,
                                 ordering='var_lbl',
                                 offset=False,
                                 save_correction=False,
                                 force_reduction=False,
                                 power_max=None,
                                 blended_lines=1,
                                 mask_rejection=True,
                                 cross_validation=True,
                                 cv_percent_rm=20,
                                 cv_sim=100,
                                 cv_frac_affected=0.01,
                                 treshold_percent=treshold_percent,
                                 snr_min = 1)       
        
        if False:
            somme = []
            for j in range(1,5):
                loc = np.argmax(abs(sts.shell_coeff[:,j]))
                si = np.sign(sts.shell_coeff[loc,j])
                somme.append(sts.shell_coeff[:,j]*si)
            somme = np.array(somme)
            somme = np.sum(somme,axis=0)
            
            spectrum_removed = ~myf.rm_outliers(somme)[0]
            sts.supress_time_spectra(liste=spectrum_removed)
    
        sts.yarara_obs_info(kw=['shell_fitted',sts.shell_composite.y])
        
        if treshold_percent<=100:
            nb_comp_shell =  myf.first_transgression(sts.shell_cv_percent,treshold_percent,relation=1)
            print('\n [AUTO] Nb shells selected by cross-validation above %.0f %% : %.0f'%(treshold_percent,nb_comp_shell))
            if nb_comp_shell>7:
                nb_comp_shell=7
            if not nb_comp_shell:
                nb_comp_shell = 1
        
        base_shell = list(sts.shell_coeff[:,1:1+nb_comp_shell].T)
        
        kw = pd.DataFrame(np.array(base_shell).T,columns=['shell_v%.0f'%(j+1) for j in range(0,len(base_shell))])
        sts.yarara_obs_info(kw=kw)

        if DBD:
            sts.lbl_fit_vec('matching_shell', sub_dico = 'matching_mad', kw_dico = 'dbd', base_vec = base_shell, time_detrending = 0, add_step = 1, save_database=False, col=0)
            sts.lbl_kolmo_cat(base_shell, sub_dico='matching_shell', kw_dico = 'dbd', alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_shell', depth_var = 'depth_rel')
            sts.lbl_xgb_cat(sub_dico='matching_shell', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
            sts.lbl_xgb_cat(sub_dico='matching_shell', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile)
            plt.close('all')

        sts.yarara_ccf_fit('matching_shell', sub_dico = 'matching_mad', proxies = base_shell, time_detrending = 0)
        sts.lbl_fit_vec('matching_shell', sub_dico = 'matching_mad', kw_dico = kw_dico, base_vec = base_shell, time_detrending = 0, add_step = 0, save_database=False)
        sts.lbl_kolmo_cat(base_shell, sub_dico='matching_shell', kw_dico = kw_dico, alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_shell', depth_var = 'depth_rel')
        sts.lbl_xgb_cat(sub_dico='matching_shell', kw_dico = kw_dico, analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        sts.lbl_xgb_cat(sub_dico='matching_shell', kw_dico = kw_dico, analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        plt.close('all')

        base_tot = base_tot + base_shell

        sts.lbl_fit_vec('matching_shell', sub_dico = 'matching_mad', kw_dico = kw_dico, base_vec = base_tot, time_detrending = 0, add_step = 1, save_database=False)        
        sts.lbl_fit_base('matching_shell', sub_dico='matching_mad', kw_dico='lbl_iter', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=1)
        sts.lbl_fit_base('matching_shell', sub_dico='matching_mad', kw_dico='lbl', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=1)
        plt.close('all')
            
        sts.simu['shell'] = sts.planet_fit_base(base_tot, time_detrending=0, substract_rv=True, p=nb_period)
        sub_dico1='matching_shell'
        
    if nb_comp_color:
        
        # =============================================================================
        # long wavelength contamination
        # =============================================================================
                
        sts.lbl_pca(reduction='pca', sub_dico=sub_dico1, kw_dico=kw_dico, col=0, nb_comp_kept='auto', ordering='var_lbl',
                    ext='_color', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=4, depth_bins=0,
                    cross_validation=True, cv_percent_rm=20, cv_sim=100, cv_frac_affected=0.01, nb_comp=10, snr_min=0.5)    
        
        if treshold_percent<=100:
            nb_comp_color =  myf.first_transgression(sts.base_cv_percent,treshold_percent,relation=1)
            print('\n [AUTO] Nb PCA vec selected by cross-validation above %.0f %% : %.0f'%(treshold_percent,nb_comp_color))
            if nb_comp_color>7:
                nb_comp_color=7
            if not nb_comp_color:
                nb_comp_color = 1
        
        base = [sts.base_vec_pca[:,j] for j in range(nb_comp_color)]
        
        kw = pd.DataFrame(np.array(base).T,columns=['pca_color_v%.0f'%(j+1) for j in range(0,len(base))])
        
        sts.yarara_obs_info(kw=kw)
                
        if DBD:
            sts.lbl_fit_vec('matching_color', sub_dico = 'matching_mad', kw_dico = 'dbd', base_vec = base, time_detrending = 0, add_step = 1, save_database=False, col=0)
            sts.lbl_kolmo_cat(base, sub_dico='matching_color', kw_dico = 'dbd', alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_color', depth_var = 'depth_rel')
            sts.lbl_xgb_cat(sub_dico='matching_color', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
            sts.lbl_xgb_cat(sub_dico='matching_color', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile)
            plt.close('all')
                
        sts.yarara_ccf_fit('matching_color', sub_dico = sub_dico1, proxies = base, time_detrending = 0)
        sts.lbl_fit_vec('matching_color', sub_dico = sub_dico1, kw_dico = kw_dico, base_vec = base, time_detrending = 0, add_step = 0, save_database=False)
        sts.lbl_kolmo_cat(base, sub_dico='matching_color', kw_dico = kw_dico, alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_color', depth_var = 'depth_rel')
        sts.lbl_xgb_cat(sub_dico='matching_color', kw_dico=kw_dico, analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        sts.lbl_xgb_cat(sub_dico='matching_color', kw_dico=kw_dico, analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        plt.close('all')
        
        #base = base + ['CaII',sub_dico.split('_')[1].upper()]
        base_tot = base_tot + base
        sts.lbl_fit_vec('matching_color', sub_dico = 'matching_mad', kw_dico = kw_dico, base_vec = base_tot, time_detrending = 0, add_step = 2)
        sts.lbl_fit_base('matching_color', sub_dico='matching_mad', kw_dico = 'lbl', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=2)
        sts.lbl_fit_base('matching_color', sub_dico='matching_mad', kw_dico = 'lbl_iter', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=2)
        plt.close('all')
    
        sts.simu['color'] = sts.planet_fit_base(base_tot, time_detrending=0, substract_rv=True, p=nb_period)
        
        sub_dico1 = 'matching_color'
    
    if nb_comp_pca:
        
        # =============================================================================
        # final correction 
        # =============================================================================
        
        sts.lbl_pca(reduction='pca', sub_dico=sub_dico1, kw_dico=kw_dico, col=0, nb_comp_kept='auto', ordering='var_lbl',
                    ext='_empca', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=0, depth_bins=0,   
                    cross_validation=True, cv_percent_rm=20, cv_sim=100, nb_comp=10, snr_min=0.5)   
        
        if treshold_percent<=100:
            nb_comp_pca =  myf.first_transgression(sts.base_cv_percent,treshold_percent,relation=1)
            print('\n [AUTO] Nb PCA vec selected by cross-validation above %.0f %% : %.0f'%(treshold_percent,nb_comp_pca))
            if nb_comp_pca>7:
                nb_comp_pca = 7
            if not nb_comp_pca:
                nb_comp_pca = 1
        
        base2 = [sts.base_vec_pca[:,j] for j in range(nb_comp_pca)]
                
        kw = pd.DataFrame(np.array(base2).T,columns=['pca_v%.0f'%(j+1) for j in range(0,len(base2))])
                
        sts.yarara_obs_info(kw=kw)
            
        if DBD:
            sts.lbl_fit_vec('matching_empca', sub_dico = 'matching_mad', kw_dico = 'dbd', base_vec = base2, time_detrending = 0, add_step = 1, save_database=False, col=0)
            sts.lbl_kolmo_cat(base2, sub_dico='matching_empca', kw_dico = 'dbd', alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_empca', depth_var = 'depth_rel')
            sts.lbl_xgb_cat(sub_dico='matching_empca', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
            sts.lbl_xgb_cat(sub_dico='matching_empca', kw_dico = 'dbd', analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile)
            plt.close('all')
                    
        sts.yarara_ccf_fit('matching_empca', sub_dico = sub_dico1, proxies = base2, time_detrending = 0)
        sts.lbl_fit_vec('matching_empca', sub_dico = sub_dico1, kw_dico = kw_dico, base_vec = base2, time_detrending = 0, add_step = 0, save_database=False)
        sts.lbl_kolmo_cat(base2, sub_dico='matching_empca', kw_dico = kw_dico, alpha_kolmo=alpha_crit, z_lim = z_crit, ext='matching_empca', depth_var = 'depth_rel')
        sts.lbl_xgb_cat(sub_dico='matching_empca', kw_dico=kw_dico, analysis='morpho-atomic-pixel', var='r', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        sts.lbl_xgb_cat(sub_dico='matching_empca', kw_dico=kw_dico, analysis='morpho-atomic-pixel', var='s', percentile=20, Plot=True, my_dpi=192, min_r_percentile=min_r_percentile) 
        plt.close('all')
        
        base_tot = base_tot + base2
        sts.lbl_fit_vec('matching_empca', sub_dico = 'matching_mad', kw_dico = kw_dico, base_vec = base_tot, time_detrending = 0, add_step = 3)
        sts.lbl_fit_base('matching_empca', sub_dico='matching_mad', kw_dico='lbl', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=3)
        sts.lbl_fit_base('matching_empca', sub_dico='matching_mad', kw_dico='lbl_iter', proxies = base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=3)
        plt.close('all')
    
        sts.simu['empca'] = sts.planet_fit_base(base_tot, time_detrending=0, substract_rv=True, p=nb_period)  

        sub_dico1 = 'matching_empca'
    
    # =============================================================================
    # BIS correction 
    # =============================================================================

    sts.lbl_treshold_blue(sub_dicos=['matching_diff','matching_morpho','matching_empca'],wave_bins=10)

    sts.import_proxies() ; bis = sts.bis.y
    base_tot = base_tot + [bis]
    
    sts.yarara_ccf_fit('matching_bis', sub_dico = sub_dico1, proxies = base_tot, time_detrending = 0)
    sts.lbl_fit_vec('matching_bis', sub_dico = 'matching_mad', kw_dico = kw_dico, base_vec = base_tot, time_detrending = 0, add_step = 4)
    sts.lbl_fit_base('matching_bis', sub_dico='matching_mad', proxies=base_tot + ['dbd_0','aba_6','aba_7'], time_detrending=0, add_step=4)
    
    sts.simu['bis'] = sts.planet_fit_base(base_tot, time_detrending=0, substract_rv=True, p=nb_period)  

    sts.planet_simu_absorption()
    
    sts.yarara_stellar_rotation(windows=100, min_nb_points=25, prot_max=100, rhk=1, kernel_rhk=0, ca2=0, ha=1, h=0, wb=1, mg1=0, nad=0, bis=1, cb=1, rv=1)
    sts.yarara_star_info(Pmag=['YARARA',np.round(sts.Pmag,1)])

    sts.yarara_curve_rv(poly_deg=0, metric='gaussian')

    if close_figure:
        plt.close('all')

    get_time_step('matching_empca')                   
    
    stage = break_func(stage) 


# =============================================================================
# PERIODOGRAM L1
# =============================================================================

if stage==30:
    sts.import_table()
    v2_dico = 'matching_empca'
    
    mask2 = sts.yarara_get_best_mask(sub_dico=v2_dico,poly_deg=2)
    mask1 = sts.yarara_get_best_mask(sub_dico=['matching_mad','matching_morpho'],poly_deg=2)

    sb_dico = ['matching_mad','matching_morpho'][int(mask1[0]=='L')]
    
    vec_shell = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_shell','rv')    
    vec_color = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_color','rv')    
    vec_lbl2 = sts.import_ccf_timeseries(mask2,v2_dico,'rv')
    vec_lbl = sts.import_ccf_timeseries(mask1,['matching_mad','matching_morpho'][int(mask1[0]=='L')],'rv')
    vec_ref = sts.import_dace_sts()

    for v in [vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]:
        v.species_recenter(species=sts.table['ins'],ref=0) 
    
    if sts.starname!='Sun':
        model_prefitted = np.array(sts.table['rv_shift'])*1000
        for i,v in enumerate([vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
            v.y += model_prefitted*(-1)*int(i==0)

    # plt.figure(figsize=(16,8))
    # for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl'],[vec_ref, vec_lbl, vec_lbl2]):
    #     sts.yarara_binned_rms_curve(v,label=name) 
    # plt.legend()
            
    sts.yarara_stellar_rotation(windows=100, min_nb_points=25, prot_max=100, rhk=1, ca2=0, ha=1, h=0, wb=1, mg1=0, nad=0, bis=1, cb=1, rv=vec_lbl2)
    
    vec_lbl2.rms_w()

    for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.periodogram_l1(v, name_ext='_'+name, photon_noise=np.min([1.3,0.75*np.round(vec_lbl2.rms,1)]), sort_val='period', p_min=2, fap_min=-2, species=sts.table['ins'])
        plt.close('all')

    signal_periods = np.array(vec_lbl2.l1_table['period'])
    if sts.starname=='Sun':
        if len(signal_periods)>5:
            signal_periods = signal_periods[0:5]
    
    # if list(vec_lbl2.l1_table['period']) != []:
    #     for dico, name, v in zip([sb_dico,'matching_empca','matching_shell','matching_shell7','matching_color'],['yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_shell_all','vec_color'],[vec_lbl, vec_lbl2, vec_shell, vec_shell_all, vec_color]):
    #         sts.keplerian_phase_test(v, sub_dico=dico, kw_dico='lbl_iter', nb_planet=3, nb_cut=3, deg=0, deg_detrend=0, photon_noise=0.7, name_ext='_l1_'+name, periods=signal_periods)  

    sts.keplerian_clean()
    
    for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, fap=1, name_ext='_'+name, deg=1, p_min=2, periods=list(v.l1_table['period']), name_pre='l1_', fit_ecc=True, species=sts.table['ins'], periodic=0.25)
    
    myf.pickle_dump(sts.planet_fitted, open(sts.dir_root+'KEPLERIAN/Planets_fitted_table_l1.p','wb'))
    
    sts.keplerian_draw_model(name_ext='_l1')
    
    sts.yarara_plot_comp_dace_v2(mask1, 
                                 sub_dico=['matching_mad','matching_morpho'][int(mask1[0]=='L')], 
                                 photon_noise=0.7,
                                 zorder1=3,
                                 zorder2=3,
                                 m_out=2)
    
    sts.yarara_plot_comp_dace_v3(mask2,
                                 sub_dico1=['matching_mad','matching_morpho'][int(mask2[0]=='L')], 
                                 sub_dico2=v2_dico,
                                 photon_noise=0.7,
                                 zorder1=3,
                                 zorder2=3,
                                 m_out=2,
                                 reference='DRS')
    
    if close_figure:
        plt.close('all')
    
    get_time_step('l1_perio_v1')                   
    
    stage = break_func(stage)


if stage==31:
    sts.import_table()
    sts.import_lbl_iter()
    
    v2_dico = 'matching_empca'
    mask_lines_clean = np.array(sts.lbl_iter[v2_dico]['catalog']['qc_telluric']).astype('bool')
    
    mask2 = sts.yarara_get_best_mask(sub_dico=v2_dico,poly_deg=2)
    mask1 = sts.yarara_get_best_mask(sub_dico=['matching_mad','matching_morpho'],poly_deg=2)

    sb_dico = ['matching_mad','matching_morpho'][int(mask1[0]=='L')]

    vec_shell = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_shell','rv')    
    vec_color = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_color','rv')    
    vec_lbl2 = sts.import_ccf_timeseries(mask2,v2_dico,'rv')
    vec_lbl = sts.import_ccf_timeseries(mask1,['matching_mad','matching_morpho'][int(mask1[0]=='L')],'rv')
    vec_ref = sts.import_dace_sts()
    
    vec_clean = sts.lbl_subselection(sub_dico=v2_dico,kw_dico='lbl_iter',mask=mask_lines_clean,valid_lines=False)

    for v in [vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]:
        v.species_recenter(species=sts.table['ins'],ref=0) 
    
    if sts.starname!='Sun':
        model_prefitted = np.array(sts.table['rv_shift'])*1000
        for i,v in enumerate([vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
            v.y += model_prefitted*(-1)*int(i==0)
    
    sts.keplerian_clean()

    for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, 
                            fap=10, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=False, species=sts.table['ins'], 
                            periodic=0.25)

    vec_lbl2.planet_fitted = sts.planet_fitted['yarara_v2_lbl'].sort_values(by='k',ascending=False)
    signal_periods = np.array(vec_lbl2.planet_fitted['p'])

    sts.planet_simu_absorption(planet=signal_periods)

    # if len(signal_periods):
    #     for dico, name, v in zip([sb_dico,'matching_empca','matching_shell','matching_shell7','matching_color'],['yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_shell_all','vec_color'],[vec_lbl, vec_lbl2, vec_shell, vec_shell_all, vec_color]):
    #         sts.keplerian_phase_test(v, sub_dico=dico, kw_dico='lbl_iter', nb_planet=3, nb_cut=3, deg=0, deg_detrend=0, photon_noise=0.7, name_ext='_'+name, periods=signal_periods)  
    
    myf.pickle_dump(sts.planet_fitted, open(sts.dir_root+'KEPLERIAN/Planets_fitted_table_iter.p','wb'))
    
    sts.keplerian_draw_model()
            
    for name,v in zip(['_v1','_v2','_shell','_color'],[vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.export_to_dace(v,ext=name)

    sts.import_dace_summary(bin_length=bin_length)

    if sts.planet:
        sts.keplerian_clean()
        file_test = sts.import_spectrum()
        true_p = file_test['parameters']['planet_injected']['period']
        
        for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
            sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, fap=1, name_ext='_'+name, deg=0, p_min=2, periods=list(true_p), name_pre='injected_', fit_ecc=False, species=sts.table['ins'], periodic=0.25)

        sts.yarara_periodogram_map(mask=None, p_min=0.7, ofac=10, planet=true_p, alias = None, period_split=7, fap=1, photon_noise=0.3, vmax=1, plot_1day=False)
        
        sts.yarara_keplerian_chain(mask2, time_detrending=0) #auto detection of p in the function                         

    sts.yarara_plot_rcorr(vec=vec_lbl2, vec_name='RV_YARARA_V2', bin_length=1, detrend=2, vmin=0.3)

    sts.yarara_plot_all_rv('LBL_kitcat_mask_'+sts.starname, v2_dico, ofac=10, photon_noise=0.7, deg_detrending=2, p_min=0, method='tree')
    
    sts.lbl_treshold_blue(sub_dicos=['matching_diff','matching_morpho',v2_dico])
    for sb,name in zip(['matching_diff','matching_morpho','matching_empca'],['_YARARA_V0','_YARARA_V1','_YARARA_V2']):
        sts.lbl_color_investigation(sub_dico=sb, kw_dico='lbl_iter', col=0, m=2, contam_training=True, wave_bins=5, ext=name)
    
    if close_figure:
        plt.close('all')

    get_time_step('fit_keplerian')                   

    stage = break_func(stage)


if stage==32:
    sts.yarara_supress_dico('matching_fourier',only_continuum=True)
    sts.yarara_supress_dico('matching_telluric',only_continuum=True)
    sts.yarara_supress_dico('matching_oxygen',only_continuum=True)
    sts.yarara_supress_dico('matching_stitching',only_continuum=True)    
    sts.yarara_supress_dico('matching_ghost_a',only_continuum=True)
    sts.yarara_supress_dico('matching_ghost_b',only_continuum=True)    
    sts.yarara_supress_dico('matching_thar',only_continuum=True)    
    sts.yarara_supress_dico('matching_smooth',only_continuum=True)    

    stage = break_func(stage)

# =============================================================================
# PRODUCE QC FILE YARARA
# =============================================================================

if stage==33:  
    try:
        os.system('touch '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/yarara_finished.txt')
    except:
        print(' [WARNING] Cannot save the qc file yarara \n')
    
# =============================================================================
# MOON CONTAMINATION (IN DEV)
# =============================================================================

if stage==34:
    #27.318678820987667
    sts.yarara_get_moon_rv(obs_loc)
    sts.import_table()
    moon2 = np.array(sts.table.rv_moon)
    berv = np.array(sts.table.berv)
    sts.yarara_time_variations(wave_min=5500,wave_max=5600,sub_dico='matching_thar',berv_shift=moon2)
    
if stage==35: #form the database for the slice master production
    nb_comp_color = 5
    fap = 0.1
    kw_dico = 'lbl_iter'
    sub_dico1, sub_dico2 = [('matching_shell','matching_color'),('matching_color','matching_empca')][1]
    alpha_crit = 1 #1% of significance alpha
    z_crit = 0.25
    
    if sub_dico2=='matching_color':
        sts.lbl_pca(reduction='pca', sub_dico=sub_dico1, kw_dico=kw_dico, col=0, nb_comp_kept=nb_comp_color, ordering='var_lbl',
                    ext='_color', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=4, depth_bins=0)    
    else:
        sts.lbl_pca(reduction='empca', sub_dico=sub_dico1, kw_dico=kw_dico, col=0, nb_comp_kept=nb_comp_color, ordering='var_lbl',
                ext='_empca', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=0, depth_bins=0)    
    
        
    base = [sts.base_vec_pca[:,j] for j in range(nb_comp_color)]
    
    for j in range(0,len(base)):
        sts.yarara_obs_info(kw=['pca_color_v%.0f'%(j+1),base[j]])
            
    sts.yarara_ccf_fit(sub_dico2, sub_dico = sub_dico1, proxies = base, time_detrending = 0)
    sts.lbl_fit_vec(sub_dico2, sub_dico = sub_dico1, kw_dico = kw_dico, base_vec = base, time_detrending = 0, add_step = 0, save_database=True)
    sts.lbl_kolmo_cat(base, sub_dico=sub_dico2, kw_dico = kw_dico, alpha_kolmo=alpha_crit, z_lim = z_crit, ext=sub_dico2, depth_var = 'depth_rel')
    sts.lbl_xgb_cat(sub_dico=sub_dico2, kw_dico=kw_dico, analysis='all', var='r', percentile=20, Plot=True, my_dpi=192) 
    sts.lbl_xgb_cat(sub_dico=sub_dico2, kw_dico=kw_dico, analysis='all', var='s', percentile=20, Plot=True, my_dpi=192) 
    plt.close('all')

    vec_color = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,sub_dico1,'rv')    
    vec_color.fit_base(sts.base_vec_pca.T)
    
    components = [myc.tableXY(sts.table.jdb,i,j) for i,j in zip(vec_color.coeff_fitted[:,np.newaxis]*sts.base_vec_pca.T,vec_color.coeff_fitted_std[:,np.newaxis]*np.ones((nb_comp_color,len(vec_color.x))))]
    plt.figure(figsize=(18,9))
    for i in np.arange(nb_comp_color):
        plt.subplot(2,nb_comp_color,i+1+nb_comp_color)
        p = np.hstack([sts.matrix_corr_lbl['pixels_l1'],sts.matrix_corr_lbl['pixels_l2']])
        w = np.hstack([sts.matrix_corr_lbl['wave'],sts.matrix_corr_lbl['wave']])
        c = np.hstack([sts.matrix_corr_lbl['s_proxy_%.0f'%(i+1)],sts.matrix_corr_lbl['s_proxy_%.0f'%(i+1)]])
        r = np.hstack([sts.matrix_corr_lbl['r_proxy_%.0f'%(i+1)],sts.matrix_corr_lbl['r_proxy_%.0f'%(i+1)]])
        c-=np.nanmedian(c)
        c/=np.nanstd(c)
        
        m = p!=0
        p = p[m] ; w = w[m] ; c = c[m] ; r = r[m]
        order = np.argsort(abs(r)) ; p = p[order] ; w = w[order] ; c = c[order] ; r = r[order]
        lim = np.nanmax([abs(np.nanpercentile(c,2.5)),np.nanpercentile(c,97.5)])
        plt.scatter(p,w,c=c,cmap='seismic',vmin=-2,vmax=2,alpha=0.8)
        ax = plt.colorbar(pad=0)
        ax.ax.set_ylabel(r'Z score $(c_%s)$'%(i+1),fontsize=15)
        
        plt.xlabel('Pixels',fontsize=15)
        if not i:
            plt.ylabel(r'Wavelength $\lambda$ [$\AA$]',fontsize=15)
        for j in [512*i for i in range(1,8)]:
            plt.axvline(x=j,color='k',ls='--',alpha=1)
        plt.axhline(y=5310,color='k',ls='-',alpha=1)
        plt.subplot(4,nb_comp_color,i+1+nb_comp_color)
        components[i].periodogram(Norm=True,level=1-fap/100) 
        plt.ylim(0,None)
        plt.subplot(4,nb_comp_color,i+1)
        plt.title('rms : %.0f cm/s'%(np.std(components[i].y)*100),fontsize=14)
        components[i].null()
        components[i].y*=100
        components[i].plot(capsize=0)
        myf.plot_TH_changed(color='r',ls='--',time=vec_color.x)
        plt.xlim(np.min(vec_color.x)-40,np.max(vec_color.x)+40)
        plt.xlabel('Time - 2,400,000 [days]',fontsize=15)
        plt.ylabel(r'$<c_%.0f>\cdot P_%.0f$ [cm/s]'%(i+1,i+1),fontsize=15)
    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.07,hspace=0.4,wspace=0.45)
    plt.savefig('/Users/cretignier/Documents/LaTeX/LaTex_cen_B_activity/PCA_%s_%s.png'%(sub_dico2,sts.starname))


# =============================================================================
# MERGE INSTRUMENT
# =============================================================================

if stage==36:
    sts.yarara_plot_all_rv('LBL_kitcat_mask_'+sts.starname, 'matching_mad', ofac=10, photon_noise=0.3)
    plt.close('all')


if stage==37:
    sts.snr_statistic()
    sts.dace_statistic()

if stage==38:
    #activity_indicators
    sts.yarara_stellar_atmos(sub_dico='matching_diff', reference='master', continuum='linear')
    sts.yarara_correct_continuum_absorption(model=None, T=None, g=None)
    sts.yarara_activity_index(sub_dico='matching_pca', continuum = 'linear', substract_map=['ghost_a','ghost_b','thar','mad','stitching','smooth'])
    
if stage==39:
    sts.ccf_order_per_order('kitcat_mask_'+sts.starname+'.p',sub_dico='matching_diff',color=0)
    sts.ccf_order_per_order(sts.mask_harps,sub_dico='matching_diff',color=1)
    sts.ccf_order_per_order('kitcat_mask_'+sts.starname+'.p',sub_dico='matching_mad',color=2) 
    plt.show()
    plt.savefig(sts.dir_root+'IMAGES/CCF_order_by_order.png')
    
if stage==40:
    sts.yarara_stellar_rotation(windows=100, min_nb_points=25, prot_max=100, rhk=1, ca2=0, ha=1, h=0, wb=1, mg1=0, nad=0, bis=1, rv=1)
    sts.yarara_star_info(Pmag=['YARARA',np.round(sts.Pmag,1)])
    #sts.yarara_curve_rv(poly_deg=2, metric='gaussian')

if stage==41:
    if len(glob.glob(sts.dir_root+'WORKSPACE/RASSINE*.p')):
        sts.yarara_analyse_summary(rm_old=True)
    
if stage==42:
    sts.yarara_merge_instrument(instruments=['HARPS03','HARPS15','HARPN','ESPRESSO18','ESPRESSO19','CORALIE14','CARMENES'],main_ins=None)
    
if stage==43:
    vec_lbl = sts.import_ccf_timeseries('LBL_kitcat_mask_'+sts.starname,'matching_morpho','rv')
    vec_lbl2 = sts.import_ccf_timeseries('LBL_kitcat_mask_'+sts.starname,'matching_empca','rv')
    sts.export_to_dace(vec_lbl, ext='_v1')    
    sts.export_to_dace(vec_lbl2, ext='_v2')    

if stage==44:
    sts.lbl_fit_sinus_film(period = myf.my_ruler(10,55,0.1,0.25), frequency = np.linspace(1/400,1/10,2000), 
                   sub_dico=['matching_pca'], seasons=[0], good_morpho=False, kw_dico='lbl', col=0,
                   color_axis='line_depth', radial_axis='r_corr', cmax=1, cmin=0,  bbox=(0.15, -0.2),
                   fit_lbl=True, loop=0, delay=10, planet = [0,26,np.pi],circle=[0.0,'r','-',2], hist=72, kde=False)

    sts.lbl_fit_sinus_film(period = myf.my_ruler(10,55,0.1,0.25), frequency = np.linspace(1/400,1/10,2000), 
                   sub_dico=['matching_pca'], seasons=[0], good_morpho=False, kw_dico='dbd', col=0,
                   color_axis='line_depth', radial_axis='r_corr', cmax=1, cmin=0,  bbox=(0.15, -0.2),
                   fit_lbl=True, loop=0, delay=10, planet = [0,26,np.pi],circle=[0.0,'r','-',2], hist=72, kde=False)    
    
    
if stage==45:
    sub_dico=sub_dico_to_analyse
    
    sts.import_table()
    sts.import_lbl_iter()
    valid_lines = np.array(sts.lbl_iter['matching_morpho']['catalog']['valid'])
    deg=0
    
    if (np.max(sts.table.jdb) - np.min(sts.table.jdb))>(2*365.25):
        fig = plt.figure(figsize=(16,7))
        mat_2,dust = sts.lbl_fit_sinus(365.25,sub_dico=sub_dico,kw_dico='lbl_iter',subplot_name=[fig,122],season=0, cmax=10, color_axis='K', deg=deg, plot_proxies=False, kde=False, light_title='After YARARA', legend=False, valid_lines=valid_lines, fontsize=16, pfont=13, rfont=13)
        plt.subplots_adjust(left=0.04,right=0.97,top=0.89,bottom=0.10)
        plt.close()
        
        mat_2['star'] = sts.starname
        mat_2['instrument'] = sts.instrument    
        
        file = sts.directory.split('Yarara/')[0]+'Yarara/statistics/'
        if not os.path.exists(file):
            os.system('mkdir '+file)
        
        if not os.path.exists(file+'1year_'+sub_dico+'.csv'):
            table2 = mat_2
        else:
            table2 = pd.read_csv(file+'1year_'+sub_dico+'.csv',index_col=0)
            loc = np.where((table2['star']==sts.starname)&(table2['instrument']==sts.instrument))[0]
            if len(loc):
                kept = np.setdiff1d(np.arange(len(table2)),loc)
                table2 = table2.loc[kept].reset_index(drop=True)
            
            table2 = pd.concat([table2,mat_2])
            
        table2 = table2.reset_index(drop=True)
        table2.to_csv(file+'1year_'+sub_dico+'.csv')
    
if stage==46:
    sts.yarara_kernel_caii(contam=True, mask='CaII_HD128621', power_snr=None, noise_kernel='unique', wave_max=None, doppler_free=False)
    sts.yarara_kernel_wb(contam=False, mask='WB_HD128621', power_snr=None, noise_kernel='unique', wave_max=None, doppler_free=True)        
   
if stage==69:
    sts.pickle_protocol(to_protocol=3) 
   
if stage==72:
    vec_lbl2 = sts.import_ccf_timeseries('LBL_kitcat_mask_'+sts.starname,'matching_empca','rv')
    vec_ref = sts.import_dace_sts()
    
    sts.simu_detection_limit2(vec_lbl2, ext='_yarara_v2_lbl', photon_noise = 0.7, degree_detrending=2,
                            nb_phase = 36, fap=0.99,
                            k = np.arange(0.2, 5.1, 0.2), 
                            p = myf.my_ruler(2.9, 2000, 0.1, 50),
                            tresh_diff=0.1, tresh_phase=0.2,
                            button_phi = 1, button_k = 1, highest=False)   

    sts.simu_detection_limit2(vec_ref, ext='_drs', photon_noise = 0.7, degree_detrending=2,
                            nb_phase = 36, fap=0.99,
                            k = np.arange(0.2, 5.1, 0.2), 
                            p = myf.my_ruler(2.9, 2000, 0.1, 50),
                            tresh_diff=0.1, tresh_phase=0.2,
                            button_phi = 1, button_k = 1, highest=False)   
    
    sts.plot_detection_limit2(file_ref=None, file=None, Mstar=None, overfit=3, color_line='k',cmap='viridis')
    
if stage==100:
    sts.import_table()
    sts.import_star_info()
    sts.import_telluric()
    sts.star_spec = sts.spectrum(norm=True,num=np.argmax(sts.table.snr))
    sts.file_test = sts.import_spectrum(num=np.argmax(sts.table.snr))
    try:
        sts.import_kitcat()
    except:
        pass
        
# =============================================================================
# SAVE TIME
# =============================================================================

myr.print_iter(time.time() - begin)

if button:
    table_time = pd.DataFrame(time_step.values(),index=time_step.keys(),columns=['time_step'])
    table_time['frac_time']=100*table_time['time_step']/np.sum(table_time['time_step'])
    table_time['time_step'] /= 60 #convert in minutes
    filename_time = sts.dir_root+'REDUCTION_INFO/Time_informations_reduction_%s.csv'%(time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()))
    table_time.to_csv(filename_time)
    
    myf.make_sound('Ya ra ra has finished')    
    


if False:     
    # =============================================================================
    # MORE analysis
    # =============================================================================
    
    sts.lbl_pca(reduction='pca', sub_dico='matching_shell', kw_dico='lbl_iter', col=0, nb_comp_kept=5, ordering='var_lbl',
            ext='_color', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=0, depth_bins=0, kernel=1)
    
    sts.lbl_pca(reduction='pca', sub_dico='matching_shell', kw_dico='lbl_iter', col=0, nb_comp_kept=5, ordering='var_lbl',
            ext='_color', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=4, depth_bins=0, kernel=0)

    sts.lbl_pca(reduction='pca', sub_dico='matching_shell', kw_dico='lbl_iter', col=0, nb_comp_kept=5, ordering='var_lbl',
            ext='_color', color_residues='k', contam_training=True, recenter=True, standardize=True, wave_bins=0, depth_bins=0, kernel=0)
    
    # =============================================================================
    #     
    # =============================================================================
        
    os.system('run trigger_yarara_harps03.py -b 1000 -i HARPS03 -s HD128621')
    
    save1 = sts.yarara_ccf(mask = 'kitcat_mask_'+sts.starname+'.p', plot=True, sub_dico='matching_pca', ccf_oversampling=1, rv_range = None, save=False) 
    save2 = sts.yarara_ccf(mask = sts.mask_harps, plot=True, sub_dico='matching_pca', ccf_oversampling=1, rv_range = None, save=False) 

    kernel = sts.yarara_create_kernel(sub_dico='matching_activity',wave_min=3900,power_snr=2,save=True,ext='magnetic_cycle')
    sts.yarara_kernel_from_proxy('CaII',sub_dico='matching_pca', substract_map=['stitching','ghost_a','ghost_b','thar','smooth'], r_min=0, wave_min=3900, wave_max=None, power_snr=1, smooth_corr=1)
    sts.yarara_kernel_from_proxy('WB',sub_dico='matching_smooth', substract_map=[], r_min=0, wave_min=3900, wave_max=None, power_snr=1, smooth_corr=1)
    
  
    test = sts.yarara_kernel_integral('matching_pca', 'Kernel_CaII_HD128621.txt', p_noise=1/np.inf, doppler_free=False, noise_kernel='else', power_snr=1) #best power ~ 1 according to simulations
    test2 = sts.yarara_kernel_integral('matching_smooth', 'Kernel_WB_HD128621.txt', p_noise=1/np.inf, doppler_free=False, noise_kernel='else', power_snr=2) #best power ~ 1 according to simulations
    
    
    sts.yarara_time_variations(sub_dico='matching_pca',reference='master',substract_map=['stitching','ghost_a','ghost_b','thar','smooth'],Plot=False,proxy_corr='CaII',proxy_detrending=0)
    plt.figure() ; sts.yarara_snr_contam(analysis='snr_limit',sub_dico='matching_activity')
    plt.figure() ; sts.yarara_snr_contam(analysis='snr_limit',sub_dico='matching_activity', slope_vec=sts.slope_corr*sts.std_norm_corr)    
    
    sts.yarara_time_variations(sub_dico='matching_pca',Plot=False,proxy_corr='WB')

    # =============================================================================
    #     
    # =============================================================================
    
    v2_dico = 'matching_bis'
    v1_dico = 'matching_morpho'
    
    vec_shell = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_shell','rv')    
    vec_color = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,'matching_color','rv')    
    vec_lbl2 = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,v2_dico,'rv')
    vec_lbl = sts.import_ccf_timeseries('LBL_ITER_kitcat_mask_'+sts.starname,v1_dico,'rv')
    
    sts.copy_dace_in_summary('ESPRESSO18', path_table='ALL_OBSERVATIONS/', bin_length=1, db=0) #do it is the star was not reduced by an instrument to put DRS values
    vec_ref = sts.import_dace_sts()
        
    for v in [vec_lbl, vec_lbl2, vec_shell, vec_shell_all, vec_color]:
        v.merge_diff1d(vec_ref)
    
    sts.import_table()
    species=np.array(sts.table['ins'])
    for v in [vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_shell_all, vec_color]:
        v.species_recenter(species=species,ref=0) 
    
    vec_perso = vec_lbl2.copy()
    vec_perso.yerr[sts.table.ins=='ESPRESSO18'] = vec_lbl.yerr[sts.table.ins=='ESPRESSO18']
    vec_perso.y[sts.table.ins=='ESPRESSO18'] = vec_lbl.y[sts.table.ins=='ESPRESSO18']
    vec_perso.yerr[sts.table.ins=='ESPRESSO19'] = vec_lbl.yerr[sts.table.ins=='ESPRESSO19']
    vec_perso.y[sts.table.ins=='ESPRESSO19'] = vec_lbl.y[sts.table.ins=='ESPRESSO19']
    
    for name,v in zip(['_handmade'],[vec_perso]):
        sts.export_to_dace(v,ext=name)
    
    vec_perso.yerr[sts.table.ins=='CORALIE14'] = np.sqrt(vec_perso.yerr[sts.table.ins=='CORALIE14']**2+2.5**2) 
    vec_perso.yerr[sts.table.ins=='HARPS03'] = np.sqrt(vec_perso.yerr[sts.table.ins=='HARPS03']**2+0.5**2) 
    vec_perso.yerr[sts.table.ins=='HARPS15'] = np.sqrt(vec_perso.yerr[sts.table.ins=='HARPS15']**2+0.5**2) 
    vec_perso.yerr[sts.table.ins=='ESPRESSO18'] = np.sqrt(vec_perso.yerr[sts.table.ins=='ESPRESSO18']**2+0.3**2) 
    vec_perso.yerr[sts.table.ins=='ESPRESSO19'] = np.sqrt(vec_perso.yerr[sts.table.ins=='ESPRESSO19']**2+0.3**2) 
    
    ins_rejected = ['.','HARPS03','HARPS15','CORALIE14','ESPRESSO18','ESPRESSO19'][0]
    species2 = species.copy()
    for rejected in ins_rejected:
        vec_perso.masked(np.array(species2!=rejected))
        species2=species2[np.array(species2!=rejected)]

    for name,v in zip(['yarara_perso_e0'],[vec_perso]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.5, 
                            fap=1, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=False, species=species2, 
                            periodic=0.25)
        
    for name,v in zip(['yarara_perso'],[vec_perso]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.5, 
                            fap=1, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=True, species=species2, 
                            periodic=0.25)

    
        
    #detection limit
    vec = sts.import_ccf_timeseries('LBL_kitcat_mask_'+sts.starname,'matching_CaWCaY','rv')
    vec_ref = sts.import_dace_sts()
    
    sts.simu_detection_limit(vec, rv_ref = None, 
                             add_noise = 0.5, degree_detrending = 2,
                             nb_perm = 1, ofac = 10, fap = None, multi_cpu = 4,
                             k = np.arange(0.2,2.41,0.1), 
                             p = myf.my_ruler(4.9, 249.9, 0.5, 5),
                             nb_phase = 10)
    
    sts.plot_detection_limit(files='2.4', faps=2, Mstar=1, tresh_diff=0.2, tresh_phase=0.2,
                             cmap='gnuplot', color_line='k', xscale='log', overfit=1, button_phi=1, button_k=1, highest=False)
    
        
    #lbl analysis
    
    sts.lbl_plot(kw_dico='lbl',col=0,num=2113,sub_dico=['matching_diff','matching_pca','matching_brute'])
    
    sts.lbl_fit_sinus(36, season=3, good_morpho=False, fit_lbl=True, planet=[0,26,np.pi/2], kw_dico = 'lbl', col = 0,
                      color_axis='line_depth',radial_axis='r_corr',sub_dico='matching_pca',num_sim=1,hist=72)
    
    sts.lbl_xgb_cat(sub_dico='matching_empca', kw_dico='lbl_iter', season=3, period=36, analysis='all', var='r', percentile=20, Plot=True, my_dpi=192) 
    
    sts.lbl_fit_sinus_all(365.25, season=0, good_morpho=False, kw_dico = 'lbl', col = 0, color_axis='line_depth')
    
    #sts.lbl_fit_sinus_periodogram_simu(seasons=[0], frequency = np.linspace(1/400,1/8,4000), period = myf.my_ruler(8,400,0.1,1), 
    #                              good_morpho=False, kw_dico = 'lbl', col = 0, sub_dico=['matching_pca','matching_brute','matching_CaWCaY'], num_sim=1)
    
    sts.lbl_fit_sinus_periodogram('matching_brute_0_lbl_0_pmin_3_pmax_1000.p')

    sts.lbl_fit_sinus(36.5, 
                       sub_dico='matching_pca', season=3, good_morpho=False, kw_dico='dbd', col=0, deg=0,
                       color_axis='K', radial_axis='r_corr', cmax=None, 
                       fit_lbl=True, hist=72, kde=False)    
    
    sts.lbl_fit_sinus_film(period = myf.my_ruler(10,55,0.1,0.25), frequency = np.linspace(1/400,1/10,2000), 
                       sub_dico=['matching_pca'], seasons=[3], good_morpho=False, kw_dico=['lbl','dbd'], col=0,
                       color_axis='line_depth', radial_axis='r_corr', cmax=1, cmin=0,  bbox=(0.15, -0.2),
                       fit_lbl=True, loop=0, delay=10, planet = [0,26,np.pi],circle=[0.0,'r','-',2], hist=72, kde=False)
    
    

    # planet test 
    planet_p = np.array([26.89, 36.41, 49.40]) 
    planet_k = np.array([1.5])
    
    k,p = np.meshgrid(planet_k,planet_p)
    k = np.ravel(k) ; p = np.ravel(p)
    
    
    for kp, pp in zip(k,p):
        ext = '_planet_%.0f_%.1f'%(pp,kp)
        sts.lbl_fit_sinus_periodogram_simu(seasons=[0], period = myf.my_ruler(3,400,0.01,2), planet=[kp,pp,0],
                                           kw_dico = 'lbl', col = 0, sub_dico=['matching_brute'], num_sim=1) 

    
    
    for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, 
                            fap=1, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=False, species=sts.table['ins'], 
                             periodic=0.25, transits={'11.6':[58631.6976,58643.2891,58944.3755,58955.9500,58967.5250,59037.1900],
                                                      '27.6':[58650.9018,58954.4100,59036.9900,59009.5900],
                                                      '107.5':[59009.7500],
                                                      '328':[59009.7500]})
                                                     
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, 
                            fap=1, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=False, species=sts.table['ins'], 
                             periodic=0.25, transits=[58631.6976,58643.2891,58944.3755,58955.9500,58967.5250,59037.1900])    
    
    sts.periodogram_kep(vec_lbl2, photon_noise=0.3, jitter=0.7, 
                        fap=10, name_ext='_transit', deg=2, 
                        p_min=2, fit_ecc=False, species=sts.table['ins'], 
                        periodic=0.25, transits=[56482.195])    
        
    

    for name,v in zip(['drs','yarara_v1_lbl','yarara_v2_lbl','yarara_shell','yarara_color'],[vec_ref, vec_lbl, vec_lbl2, vec_shell, vec_color]):
        sts.periodogram_kep(v, photon_noise=0.3, jitter=0.7, periods=[4.31365,6.58881],
                            fap=1, name_ext='_'+name, deg=1, 
                            p_min=2, fit_ecc=False, species=sts.table['ins'], known_planet=[4.31365,6.58881], 
                            periodic=0.25, transits={'4,31':[1685.4751+57000],
                                                      '6.5881':[1690.38814+57000]})
    