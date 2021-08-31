#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:18:53 2020

@author: Cretignier Michael 
@university University of Geneva
"""

import os 
import sys 
import getopt
from colorama import Fore
import time
import glob as glob
import pandas as pd
import numpy as np

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

fast = 0
rassine_full_auto = 1
bin_length = 1
force_reduction = 0
type_reduction = 'all'
prefit_planet = False
drs_version='old'
planet_simu = 0
m_clipping = 3

instrument =['HARPS03','HARPS15','HARPN']

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:i:a:l:f:r:t:k:D:p:c:')
    for j in optlist:
        if j[0] == '-s':
            star = j[1]
        elif j[0] == '-i':
            instrument = j[1].split(',')
        elif j[0] == '-a':
            rassine_full_auto = int(j[1]) 
        elif j[0] == '-l':
            bin_length = int(j[1])
        elif j[0] == '-f':
            fast = int(j[1]) 
        elif j[0] == '-r':
            force_reduction = int(j[1]) 
        elif j[0] == '-t':
            type_reduction = j[1]
        elif j[0] == '-k':
            prefit_planet = int(j[1])
        elif j[0] == '-D':
            drs_version = j[1]
        elif j[0] == '-p':
            planet_simu = int(j[1])
        elif j[0] == '-c':
            m_clipping = int(j[1])

if not os.path.exists(root+'/Yarara/Summary_database_reduction.csv'):
    database = pd.DataFrame({'star':'----','ins_name':'--------','qc_rassine':'----------','last_rassine':'------------','qc_yarara':'---------','last_yarara':'-----------','nb_pts':'------'},index=[0])
    database.to_csv(root+'/Yarara/Summary_database_reduction.csv')

print('YARARA will be launched with the star : %s'%(star))

if not os.path.exists(root+'/Yarara/'+star):
    os.system('mkdir '+root+'/Yarara/'+star)

if not os.path.exists(root+'/Yarara/'+star+'/data/'):
    os.system('mkdir '+root+'/Yarara/'+star+'/data/')

if not os.path.exists(root+'/Yarara/'+star+'/data/s1d/'):
    os.system('mkdir '+root+'/Yarara/'+star+'/data/s1d/')

if not os.path.exists(root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'):
    os.system('mkdir '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/')
        

# =============================================================================
# REDUCTION
# =============================================================================

#old reduction 
begin2 = 3
    
dico = {'HARPS03':['14','34'],
        'HARPS15':['14','34'],
        'HARPN':['13','32'],
        'ESPRESSO18':['13','31'],
        'ESPRESSO19':['13','31'],
        'CORALIE14':['12','26']}

for ins in instrument:
    
    name, end1, end2 = ins.lower(), dico[ins][0], dico[ins][1]
    
    if not os.path.exists(root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/'):
        os.system('mkdir '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/') 
       
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    log_file = root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/log_file_%s_%s.log'%(type_reduction,current_time)      
    
    now = time.time()

    if type_reduction=='rassine':
        os.system('rm '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/rassine_finished.txt 2>/dev/null')

        print('\nSuppression of eventual pre-existing directory : '+root+'/Yarara/'+star+'/data/s1d/'+ins+'\n')
        os.system('rm -rf '+root+'/Yarara/'+star+'/data/s1d/'+ins+' 2>/dev/null') 
        os.system('rm '+root+'/Yarara/'+star+'/data/s1d/*%s'%(ins)+'*'+' 2>/dev/null') 
        os.system('rm '+root+'/Yarara/'+star+'/data/s1d/ALL_OBSERVATIONS/*%s'%(ins)+'*'+' 2>/dev/null') 

        if (not os.path.exists(root+'/Yarara/'+star+'/data/s1d/'+ins+'/DACE_TABLE/Dace_extracted_table.csv'))|(force_reduction):
            os.system('python trigger_yarara_%s.py -b -2 -e -2 -c 0 -i %s -s %s -f %.0f -l %.0f -a %.0f -k %.0f -v -1 -D %s -m %.0f | tee -a %s'%(name, ins, star, fast, bin_length, rassine_full_auto, prefit_planet, drs_version, m_clipping, log_file))
        
        if (not os.path.exists(root+'/Yarara/'+star+'/data/s1d/'+ins+'/STACKED/'))|(force_reduction):
            os.system('python trigger_yarara_%s.py -b -1 -e 1 -c 1 -i %s -s %s -f %.0f -l %.0f -a %.0f -v 0 -D %s| tee -a %s'%(name, ins, star, fast, bin_length, rassine_full_auto, drs_version, log_file))
   
        error = os.system('mv '+root+'/Yarara/'+star+'/data/s1d/'+ins+'/rassine_finished.txt '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins)
        
    elif type_reduction=='yarara':             
        os.system('rm '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/yarara_finished.txt 2>/dev/null')

        os.system('python trigger_yarara_%s.py -b 2 -e %s -c 1 -i %s -s %s -f %.0f -l %.0f -a %.0f -v 1 -D %s -p %.0f | tee -a %s'%(name, end1, ins, star, fast, bin_length, rassine_full_auto, drs_version, planet_simu, log_file))
    
        os.system('python trigger_yarara_%s.py -b 3 -e %s -c 1 -i %s -s %s -r master -f %.0f -l %.0f -a %.0f -v 2 -D %s -p %.0f | tee -a %s'%(name, end2, ins, star, fast, bin_length, rassine_full_auto, drs_version, planet_simu, log_file))

        error = int(not os.path.exists(root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/yarara_finished.txt'))

        total_time = (time.time() - now) ; hours = total_time // 3600 % 24 ; minutes = total_time // 60 % 60 ; seconds = total_time % 60
        print(Fore.RED + " ==============================================================================\n [INFO] Total time (%s) : %.0fh %.0fm %.0fs \n ==============================================================================\n"%(ins,hours,minutes,seconds)+ Fore.WHITE)
 
        if not len(glob.glob(root+'/Yarara/'+star+'/data/s1d/'+ins+'/*')):
            os.system('rm -rf '+root+'/Yarara/'+star+'/data/s1d/'+ins)
        
        os.system('python trigger_yarara_%s.py -b 42 -e 42 -c 0 -i %s -s %s -r master -f %.0f -l %.0f -a %.0f -v 42 -D %s -p %.0f | tee -a %s'%(name, ins, star, fast, bin_length, rassine_full_auto, drs_version, planet_simu, log_file))
        
        os.system('python trigger_yarara_%s.py -i %s -s %s -b 43 -e 43 -c 0 -v 42 -D %s | tee -a %s'%(name, 'INS_merged', star, drs_version, log_file))            

        #os.system('python trigger_yarara_%s.py -b 666 -e 666 -c 0 -i %s -s %s -v 42 -D %s | tee -a %s'%(name, ins, star, drs_version, log_file))

    elif type_reduction=='transit':             
        os.system('rm '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/transit_finished.txt 2>/dev/null')

        os.system('python trigger_yarara_transit.py -b 0 -e 11 -c 1 -i %s -s %s -f 0 -l 0 -a %.0f -v 1 -D %s -p %.0f | tee -a %s'%(ins, star, rassine_full_auto, drs_version, planet_simu, log_file))
    
        error = int(not os.path.exists(root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/transit_finished.txt'))

        total_time = (time.time() - now) ; hours = total_time // 3600 % 24 ; minutes = total_time // 60 % 60 ; seconds = total_time % 60
        print(Fore.RED + " ==============================================================================\n [INFO] Total time (%s) : %.0fh %.0fm %.0fs \n ==============================================================================\n"%(ins,hours,minutes,seconds)+ Fore.WHITE)
 
        if not len(glob.glob(root+'/Yarara/'+star+'/data/s1d/'+ins+'/*')):
            os.system('rm -rf '+root+'/Yarara/'+star+'/data/s1d/'+ins)
        
    else: #to rerun typical sequence in yarara 
        
        short = type_reduction.split('yarara_')[1]
        
        if short[0]=='m':
            steps = short.split('merged_')[1]
        else:
            steps = short
        
        begin2 = steps.split('_')[0]
        try:
            end2 = steps.split('_')[1]
        except:
            pass
        
        if short[0]=='m':
            os.system('python trigger_yarara_harps03.py -b %s -e %s -c 1 -i %s -s %s -r master -f %.0f -l %.0f -a %.0f -v 2 -D %s -p %.0f'%(begin2, end2, 'INS_merged', star, fast, bin_length, rassine_full_auto, drs_version, planet_simu))
        else:
            os.system('python trigger_yarara_%s.py -b %s -e %s -c 1 -i %s -s %s -r master -f %.0f -l %.0f -a %.0f -v 2 -D %s -p %.0f'%(name, begin2, end2, ins, star, fast, bin_length, rassine_full_auto, drs_version, planet_simu))
        
        

    if not len(glob.glob(root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/*.log')):
        os.system('rm -rf '+root+'/Yarara/'+star+'/data/s1d/YARARA_LOGS/'+ins+'/')    
        
    if (type_reduction=='rassine')|(type_reduction=='yarara')|(1==1):
        database = pd.read_csv(root+'/Yarara/Summary_database_reduction.csv',index_col=0)
        database = database.reset_index(drop=True)
        index_star = database.loc[(database['star']==star)&(database['ins_name']==ins)].index
        nb_spec1 = len(glob.glob(root+'/Yarara/'+star+'/data/s1d/'+ins+'/WORKSPACE/Stacked*'))
        nb_spec2 = len(glob.glob(root+'/Yarara/'+star+'/data/s1d/'+ins+'/STACKED/Stacked*'))
        nb_spec3 = len(glob.glob(root+'/Yarara/'+star+'/data/s1d/'+ins+'/PREPROCESSED/*.p'))
        nb_pts = np.max([nb_spec1,nb_spec2,nb_spec3])
        if len(index_star):
            database.loc[index_star,'qc_'+type_reduction] = int(error==0)*int(nb_pts!=0)
            database.loc[index_star,'last_'+type_reduction] = current_time
            database.loc[index_star,'nb_pts'] = nb_pts
        else:
            new_line = pd.DataFrame({'star':star,'ins_name':ins,'qc_'+type_reduction:int(error==0),'last_'+type_reduction:current_time,'nb_pts':nb_pts},index=[database.index[-1]+1])
            database = pd.concat([database,new_line])
        database = database.sort_values(by=['star','ins_name'])
        database = database.dropna(how='all')  
        database = database.loc[database['nb_pts']!='0']
        
        all_stars = np.unique(np.array(database['star']))
        all_stars_idx = np.arange(len(all_stars))
        dicotomie = {i:j for i,j in zip(all_stars,all_stars_idx)}
        database.index = np.array([dicotomie[i] for i in database['star']])
        database.to_csv(root+'/Yarara/Summary_database_reduction.csv')

print('The reduction is finished !')





