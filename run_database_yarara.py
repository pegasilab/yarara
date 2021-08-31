#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:37:26 2020

@author: cretignier
"""

import pandas as pd 
import numpy as np
import os
import getopt
import sys
import glob as glob

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

instrument =['HARPS03','HARPS15','HARPN']
type_reduction = ''
force_reduction = 0 
prefit_planet = 0
qc_check = ''
begin = 1
bin_length = 1
end = None
liste = []
drs_version = 'old'
planet_simu = 0
m_clipping=3

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'i:t:r:b:e:q:s:k:D:l:p:c:')
    for j in optlist:
        if j[0] == '-i':
            instrument = j[1].split(',')
        if j[0] == '-t':
            type_reduction = j[1]
        if j[0] == '-r':
            force_reduction = int(j[1])
        if j[0] == '-b':
            begin = int(j[1])
        if j[0] == '-e':
            end = int(j[1])
        if j[0] == '-q':
            qc_check = j[1]
        if j[0] == '-s':
            liste = j[1].split(',')
        if j[0] == '-k':
            prefit_planet = int(j[1])
        if j[0] == '-D':
            drs_version = j[1]
        if j[0] == '-l':
            bin_length = int(j[1])
        if j[0] == '-p':
            planet_simu = int(j[1])
        elif j[0] == '-c':
            m_clipping = int(j[1])

if liste[0]=='database':
    liste = []
    for ins in instrument:
        star = np.array(pd.read_csv(root+'/Python/Dace_summary_%s.csv'%(ins),index_col=0)['name'])
        vec = [''.join(s.split(' ')) for s in star]
        liste.append([vec,[ins]*len(vec)])    
        print(' [INFO] Number of stars for %s : %.0f'%(ins,len(vec)))
    liste = np.hstack(liste).T
    stars = np.sort(np.unique(liste[:,0]))

elif liste[0]=='summary':
    liste=[]
    for ins in instrument:
        table_reduction = pd.read_csv(root+'/Yarara/Summary_database_reduction.csv')
        table_reduction['nb_pts'][0] = '0'
        nb_pts = table_reduction['nb_pts'].astype('float')
        table_reduction = table_reduction.loc[nb_pts>20]
        table_reduction = table_reduction.loc[table_reduction['ins_name']==ins]
        star = np.array(table_reduction['star'])
        vec = [''.join(s.split(' ')) for s in star]
        liste.append([vec,[ins]*len(vec)])    
        print(' [INFO] Number of stars for %s : %.0f'%(ins,len(vec)))    
    liste = np.hstack(liste).T
    stars = np.sort(np.unique(liste[:,0]))
else:
    stars = liste
    
# if not len(liste):
#     for ins in instrument:
#         if os.path.exists(root+'/Yarara/Summary_database_reduction.csv'):
#             table_reduction = pd.read_csv(root+'/Yarara/Summary_database_reduction.csv')
#             table_reduction = table_reduction.loc[table_reduction['nb_pts']!='0']
#             table_reduction = table_reduction.loc[table_reduction['ins_name']==ins]
#             star = np.array(table_reduction['star'])
#         else:
#             star = np.array(pd.read_csv(root+'/Yarara/Dace_summary_%s.csv'%(ins),delimiter='\t')['name'])
#         vec = [''.join(s.split(' ')) for s in star]
#         liste.append([vec,[ins]*len(vec)])    
#         print(' [INFO] Number of stars for %s : %.0f'%(ins,len(vec)))



if qc_check:
    print('\n [INFO] Rejecting stars qith qc = 1 for the reduction... \n')
    if os.path.exists(root+'/Yarara/Summary_database_reduction.csv'):
        table_qc = pd.read_csv(root+'/Yarara/Summary_database_reduction.csv')
        table_qc = table_qc.loc[table_qc['nb_pts']!=0]
        table_qc = table_qc[np.in1d(np.array(table_qc['ins_name']),instrument)]

        star_passed = np.array(table_qc.loc[table_qc['qc_'+qc_check]==1,'star'])
        star_not_passed = np.array(table_qc.loc[table_qc['qc_'+qc_check]==0,'star'])
        star_passed = star_passed[~np.in1d(star_passed,star_not_passed)]
        
        star_rejected = np.in1d(liste[:,0],star_passed)
        short_liste = liste[~star_rejected]
        for ins in instrument:
            print(' [INFO] Number of stars for %s : %.0f'%(ins,sum(short_liste[:,1]==ins)))
        
        stars = np.sort(np.unique(short_liste[:,0]))

    
if end is None:
    end = len(stars)

stars = stars[begin-1:end]

print(' [INFO] %s DRS output product used'%(drs_version))
print(' [INFO] The following %.0f stars will be reduced: \n \n '%(len(stars)),stars)

ins = ','.join(instrument)

if type_reduction=='rassine':
    for star in stars:
        print('sbatch run_rassine.s %s %s %.0f %.0f %.0f %s %.0f'%(star,ins,bin_length,force_reduction,prefit_planet,drs_version,m_clipping))
        os.system('sbatch run_rassine.s %s %s %.0f %.0f %.0f %s %.0f'%(star,ins,bin_length,force_reduction,prefit_planet,drs_version,m_clipping))
    
elif type_reduction=='yarara':
    for star in stars:
        print('sbatch run_yarara.s %s %s %.0f %s %.0f'%(star,ins,bin_length,drs_version,planet_simu))
        os.system('sbatch run_yarara.s %s %s %.0f %s %.0f'%(star,ins,bin_length,drs_version,planet_simu))
elif type_reduction=='transit':
    for star in stars:
        print('sbatch run_transit.s %s %s %.0f %s %.0f'%(star,ins,bin_length,drs_version,planet_simu))
        os.system('sbatch run_transit.s %s %s %.0f %s %.0f'%(star,ins,bin_length,drs_version,planet_simu))
else:
    for star in stars:
        print('sbatch run_yarara_step.s %s %s %.0f %s %s %.0f'%(star,ins,bin_length,type_reduction,drs_version,planet_simu))
        os.system('sbatch run_yarara_step.s %s %s %.0f %s %s %.0f'%(star,ins,bin_length,type_reduction,drs_version,planet_simu))
            
    
    