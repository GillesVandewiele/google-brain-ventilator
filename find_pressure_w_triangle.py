import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pqdm.processes import pqdm
import itertools
import multiprocessing
from scipy.stats import mode
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv('train.csv')

PRESSURE_MAX  = 64.82099173863328 
PRESSURE_MIN  = -1.7551400036622216
PRESSURE_STEP = 0.0703021454512
PRESSURE_MIN2 = PRESSURE_MIN + PRESSURE_STEP

posref = np.array(list(range(80)))
Kp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
Ki = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
Kt = np.array([10, 15, 20, 25, 30, 35])
predictions = {}

Ps = np.arange(PRESSURE_MIN, PRESSURE_MAX+PRESSURE_STEP, PRESSURE_STEP)
P1s = Ps
P2s = Ps[:, None]
PDiff = P1s-P2s

def match(t_, kt_range=None):
    if kt_range is None:
        kt_range = [10, 15, 20, 25, 30, 35]
    kp, ki, u_in, dt = t_
    bestN = 5
    Ppredsauv = np.zeros(80) - 999
    best_params = [-1, -1, -1]
    for kt in kt_range:
        P1List = np.zeros(32) - 999
        Ppred = np.zeros(80) - 999
        n = 0
        no_matches = 0
        for j in range(32):            
            u1 = u_in[j]
            u2 = u_in[j + 1]
            ki2 = ki * dt[j] / (0.5 + dt[j])
            found = False
            if j > 0 and P1List[j - 1] != -999:
                P1 = P1List[j - 1]
                Upred1 = u1+kp*(P1-PRESSURE_MIN)+ki2*(kt-PRESSURE_MIN-(u1-kp*(kt-P1))/ki)
                Upred2 = u1+kp*(P1-(PRESSURE_MIN + PRESSURE_STEP))+ki2*(kt-(PRESSURE_MIN + PRESSURE_STEP)-(u1-kp*(kt-P1))/ki)
                slope = (Upred2 - Upred1)
                x_intersect = (u2 - Upred1) / slope
                    
                if np.abs(x_intersect - np.round(x_intersect)) < 1e-10:
                    n += 1
                    pos3 = int(np.round(x_intersect))
                    Ppred[j+1] = PRESSURE_MIN + pos3 * PRESSURE_STEP
                    P1List[j] = PRESSURE_MIN + pos3 * PRESSURE_STEP
                    found = True
            else:

                P1_arange=np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP)
                Upred1 = u1+kp*(P1_arange-PRESSURE_MIN)+ki2*(kt-PRESSURE_MIN-(u1-kp*(kt-P1_arange))/ki)
                Upred2 = u1+kp*(P1_arange-(PRESSURE_MIN + PRESSURE_STEP))+ki2*(kt-(PRESSURE_MIN + PRESSURE_STEP)-(u1-kp*(kt-P1_arange))/ki)
                slope = (Upred2 - Upred1)
                x_intersect = (u2 - Upred1) / slope
                diff=np.abs(x_intersect - np.round(x_intersect))

                #for i in range(len(diff)):
                if diff.min() < 1e-10:
                    n += 1
                    pressure_candidates = np.round(x_intersect)[np.where(diff < 1e-10)[0]] * PRESSURE_STEP + PRESSURE_MIN
                    mean_pressure = np.mean(Ppred[Ppred != -999])
                    best_candidate = min(pressure_candidates, key=lambda x: abs(x - mean_pressure))
                    Ppred[j+1] = best_candidate
                    P1List[j] = best_candidate
                    found = True
                    
#         if n > 0:
#             print(kp, ki, kt, n)

        if n > bestN:
#             print(n, kp, ki, kt)
#             print(Ppred)
            bestN = n
            Ppredsauv = Ppred
            best_params = [kp, ki, kt]

    return bestN, Ppredsauv, best_params

def match2(t_, kt_range=None):
    if kt_range is None:
        kt_range = [10, 15, 20, 25, 30, 35]
        
    kp, ki, u_in, dt, Ppredsauv = t_
    bestN = 5
    best_params = [-1, -1, -1]
    for kt in kt_range:
        P1List = np.zeros(32) - 999
        Ppred = np.zeros(80) - 999
        PpredP1 = np.zeros(80) - 999
        n = 0
        no_matches = 0
        for j in range(1, 32):
            u1 = u_in[j - 1]
            u2 = u_in[j]
            ki2 = ki * dt[j - 1] / (0.5 + dt[j - 1])
            for P1 in np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP):
                P2 = np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP)
                #U1+kp*(P1-P2)+ki2*(kt-P2-(U1-kp*(kt-P1))/ki);
                Upred=u1+kp*(P1-P2)+ki2*(kt-P2-(u1-kp*(kt-P1))/ki)
                err=np.abs(u2-Upred);
                if np.min(err) < 1e-10:
                    n += 1
                    pos3 = np.where(err == np.min(err))[0][0]
                    if P1 == P2[pos3]:
                        Ppred[j] = P2[pos3]
#                     Ppred[j + 1] = P1
#                     PpredP1[j] = P1
#                     break

        if n > bestN:# and np.sum(Ppred[Ppred != -999] == PpredP1[Ppred != -999]) > 1:
            bestN = n
            Ppredsauv[(Ppred != -999) & (Ppredsauv == -999)] = Ppred[(Ppred != -999) & (Ppredsauv == -999)]
            # Ppredsauv[(PpredP1 != -999) & (Ppredsauv == -999)] = PpredP1[(PpredP1 != -999) & (Ppredsauv == -999)]
            best_params = [kp, ki, kt]

    return Ppredsauv

def match_triangle(t_):
    kp, ki, kt, u_in, dt, preds = t_
    
    dt2 = dt / (0.5 + dt)
    found_u2 = None
    start = np.argmax(preds[2:] == -999) + 2
    
    for j in range(start, 32):
        if preds[j - 1] == -999:
            found_u2 = None
            continue
            
        if preds[j] != -999 or preds[j + 1] != -999 or preds[j + 2] != -999:
            found_u2 = None
            continue
            
        if 0 in [u_in[j], u_in[j + 1], u_in[j + 2]] or 100 in [u_in[j], u_in[j + 1], u_in[j + 2]]:
            found_u2 = None
            continue
        
        if found_u2 is None:
            val = u_in[j - 1]
            integ1 = (val - kp * (kt - preds[j - 1])) / ki
        else:
            val = found_u2
            integ1 = (val - kp * (kt - preds[j - 1])) / ki
            
        for P2 in np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP): #
            integ2 = integ1 + dt2[j - 1] * (kt - P2 - integ1)
            u2 = kp * (kt - P2) + ki * integ2

            P3 = np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP)
            integ3 = integ2 + dt2[j] * (kt - P3 - integ2)
            u3 = kp * (kt - P3) + ki * integ3

            integ41 = integ3 + dt2[j + 1] * (kt - PRESSURE_MIN - integ3)
            u41 = kp * (kt - PRESSURE_MIN) + ki * integ41
            dU1 = [u2 - u_in[j], u3 - u_in[j + 1], u41 - u_in[j + 2]]
            x1 = ((dU1[1] - dU1[0]) / dt[j]) - ((dU1[2] - dU1[1]) / dt[j + 1])

            integ42 = integ3 + dt2[j + 1] * (kt - PRESSURE_MIN2 - integ3)
            u42 = kp * (kt - PRESSURE_MIN2) + ki * integ42
            dU2 = [u2 - u_in[j], u3 - u_in[j + 1], u42 - u_in[j + 2]]
            x2 = ((dU2[1] - dU2[0]) / dt[j]) - ((dU2[2] - dU2[1]) / dt[j + 1])

            slope = (x2 - x1)
            x_intersect = -x1 / slope

            if np.min(np.abs(x_intersect - np.round(x_intersect))) < 1e-10:
                pos3 = np.where(np.abs(x_intersect - np.round(x_intersect)) < 1e-10)[0][0]
                
                if preds[j] == -999:
                    preds[j] = P2
                elif abs(preds[j] - P2) > 1e-10:
                    continue
                    
                if preds[j + 1] == -999:
                    preds[j + 1] = P3[pos3]
                elif abs(preds[j + 1] - P3[pos3]) > 1e-10:
                    continue
                    
                if preds[j + 2] == -999:
                    preds[j + 2] = np.round(x_intersect[pos3]) * PRESSURE_STEP + PRESSURE_MIN
                    
                found_u2 = u2
                    
                break
                
    return preds

def fill_gaps_b(t_):
    kp, ki, kt, u_in, dts, preds = t_
    
    us = np.zeros(80) - 999
    
    for j in range(32):
        u1, u2 = u_in[j], u_in[j+1]
        ki2 = ki * dts[j+1] / (0.5 + dts[j+1])

        u2_hat= u1+kp*(PDiff) + ki2*(kt-P2s-(u1-kp*(kt-P1s))/ki)
        m = np.abs(u2 - u2_hat) < 1e-10
        if np.any(m):
            pos = np.where(m)[0][0]
            us[j+1] = u2
            if preds[j + 1] == -999:
                preds[j+1] = P2s[pos]
                
    return preds


def fill_gaps(t_):
    kp, ki, kt, u_in, dt, preds = t_
    for j in range(2, 32):
        if preds[j] != -999 or preds[j - 1] == -999:
            continue
            
        integ1 = (u_in[j - 1] - kp * (kt - preds[j - 1])) / ki
        for P in np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP):
            integ2 = integ1 + (dt[j - 1] / (dt[j - 1] + 0.5)) * (kt - P - integ1)
            u2 = kp * (kt - P) + ki * integ2
            if np.abs(u2 - u_in[j]) < 1e-10:
                preds[j] = P
                break
                
    return preds

def fill_second(t_):
    kp, ki, kt, u_in, dt, preds = t_
    for j in range(32, -1, -1):

        for P in np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP):
            integ1 = (u_in[j] - kp * (kt - P)) / ki

            error2 = kt - preds[j + 1]
            integ2 = integ1 + (dt[j] / (dt[j] + 0.5)) * (error2 - integ1)
            u1 = kp * error2 + ki * integ2

            error3 = kt - preds[j + 2]
            integ3 = integ2 + (dt[j + 1] / (dt[j + 1] + 0.5)) * (error3 - integ2)
            u2 = kp * error3 + ki * integ3

            error4 = kt - preds[j + 3]
            integ4 = integ3 + (dt[j + 2] / (dt[j + 2] + 0.5)) * (error4 - integ3)
            u3 = kp * error4 + ki * integ4

            if abs(u1 - u_in[j + 1]) + abs(u2 - u_in[j + 2]) + abs(u3 - u_in[j + 3]) < 1e-8 and preds[j] == -999:
                preds[j] = P
    
    return preds

def match_triangle_backwards(t_):
    kp, ki, kt, u_in, dt, preds = t_
    
    dt2 = dt / (0.5 + dt)
    found = None
    
    for j in range(32, -1, -1):
        if preds[j] != -999:
            found = None
            continue
            
        if found is None and preds[j + 3] == -999:
            continue
            
        if 0 in [u_in[j], u_in[j + 1], u_in[j + 2], u_in[j + 3]] or 100 in [u_in[j], u_in[j + 1], u_in[j + 2], u_in[j + 3]]:
            found = None
            continue

        if found is None:
            P_known = preds[j + 3]
            val = u_in[j + 3]
        else:
            P_known = P2
            val = found

        P0_found = False
        for P0 in np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP):

            P1 = np.arange(PRESSURE_MIN, PRESSURE_MAX + PRESSURE_STEP, PRESSURE_STEP)

            integ3 = (val - kp * (kt - P_known)) / ki
            integ2 = (integ3 - dt2[j + 2] * (kt - P_known))/(1 - dt2[j + 2])

            integ11 = (integ2 - dt2[j + 1] * (kt - PRESSURE_MIN))/(1 - dt2[j + 1])
            integ01 = (integ11 - dt2[j + 0] * (kt - P1))/(1 - dt2[j + 0])
            u01 = kp*(kt - P0) + ki*integ01
            u11 = kp*(kt - P1) + ki*integ11
            u21 = kp*(kt - PRESSURE_MIN) + ki*integ2
            du1 = [u01 - u_in[j], u11 - u_in[j + 1], u21 - u_in[j+2]]
            x1 = ((du1[1] - du1[0]) / dt[j]) - ((du1[2] - du1[1]) / dt[j + 1])

            integ12 = (integ2 - dt2[j + 1] * (kt - PRESSURE_MIN2))/(1 - dt2[j + 1])
            integ02 = (integ12 - dt2[j + 0] * (kt - P1))/(1 - dt2[j + 0])
            u02 = kp*(kt - P0) + ki*integ02
            u12 = kp*(kt - P1) + ki*integ12
            u22 = kp*(kt - PRESSURE_MIN2) + ki*integ2
            du2 = np.array([u02 - u_in[j], u12 - u_in[j + 1], u22 - u_in[j+2]])
            x2 = ((du2[1] - du2[0]) / dt[j]) - ((du2[2] - du2[1]) / dt[j + 1])

            slope = (x2 - x1)
            x_intersect = -x1 / slope

            if np.min(np.abs(x_intersect - np.round(x_intersect))) < 1e-10:
                pos3 = np.where(np.abs(x_intersect - np.round(x_intersect)) < 1e-10)[0][0]

                P1 = P1[pos3]
                P2 = np.round(x_intersect[pos3]) * PRESSURE_STEP + PRESSURE_MIN

                integ1 = (integ2 - dt2[j + 1] * (kt - P2))/(1 - dt2[j + 1])
                integ0 = (integ1 - dt2[j + 0] * (kt - P1))/(1 - dt2[j + 0])
                u0 = kp*(kt - P0) + ki*integ0
                u1 = kp*(kt - P1) + ki*integ1
                u2 = kp*(kt - P2) + ki*integ2

                if preds[j + 2] == -999:
                    preds[j + 2] = P2
                elif abs(preds[j + 2] - P2) > 1e-10:
                    continue

                if preds[j + 1] == -999: 
                    preds[j + 1] = P1
                elif abs(preds[j + 1] - P1) > 1e-10:
                    continue

                if preds[j] == -999:
                    preds[j] = P0

                found = u2
                P0_found = True
                break

        if not P0_found:
            found = None
            
    return preds
    
def brute(t_, best_params=None):
    
    breath_nr, breath = t_
    pos = 80*breath_nr + posref
    
    data = train[train['breath_id'] == breath]
    dt = np.diff(data['time_step'])
    u_in = data['u_in'].values
    u_out = data['u_out'].values
    
    # Construct parameter lists
    if best_params is None:
        params = itertools.product(Kp, Ki)
        params = [list(p) + [u_in, dt] for p in params]

        pool = multiprocessing.Pool(processes=1)
        results = (pool.map(match, params))
        
    else:
        results = [match([best_params[0], best_params[1], u_in, dt], kt_range=[best_params[2]])]
        results += [match([0, ki, u_in, dt]) for ki in Ki]
        
    # Do a first match with a swipe across all parameters
    all_preds = defaultdict(list)
    Ppredsauv = np.zeros(80) - 999
    for result in sorted(results, key=lambda x: -x[0]):
        
        if result[0] <= 5:
            continue
            
        Ppred = result[1]
        for i, p in enumerate(Ppred):
            if p != -999:
                all_preds[i].extend([p] * result[0])

    for i in range(32):
        if len(all_preds[i]) > 0:
            Ppredsauv[i] = mode(all_preds[i])[0][0]
        
    # Re-use the best parameters for the matchers to come, on ties do not chose kp=0
    result = max(results, key=lambda x: x[0] - 0.01*(x[2][0] == 0))
    
    if result[2][0] == 0:
        for j in range(32, 0, -1):
            if Ppredsauv[j - 1] == -999:
                Ppredsauv[j] = -999
                
        if Ppredsauv[0] != -999:
            Ppredsauv[0] = -999
    
    Ppredsauv = match2((result[2][0], result[2][1], u_in, dt, Ppredsauv), kt_range=[result[2][2]])
    Ppredsauv = match_triangle((result[2][0], result[2][1], result[2][2], u_in, dt, Ppredsauv))
    
    Ppredsauv = fill_gaps((result[2][0], result[2][1], result[2][2], u_in, dt, Ppredsauv))
    Ppredsauv = fill_second((result[2][0], result[2][1], result[2][2], u_in, dt, Ppredsauv))
    
    old = np.sum(Ppredsauv != -999)
    Ppredsauv = match_triangle_backwards((result[2][0], result[2][1], result[2][2], u_in, dt, Ppredsauv))
    Ppredsauv = fill_gaps_b((result[2][0], result[2][1], result[2][2], u_in, dt, Ppredsauv))
    
    Ppredsauv[u_out == 1] = -999
    Ppredsauv[(u_in == 0) | (u_in == 100)] = -999
    
    if result[2][0] == 0:
        for j in range(32, 0, -1):
            if Ppredsauv[j - 1] == -999:
                Ppredsauv[j] = -999
                
        if Ppredsauv[0] != -999:
            Ppredsauv[0] = -999
                
    print(f'breath: {breath}\tnr matches = {np.sum(Ppredsauv != -999)}\t\t'
          f'parameters: {result[2]}\tfirst match ix: {np.argmax(Ppredsauv != -999)}')

    result = (np.sum(Ppredsauv != -999), Ppredsauv, result[2])
    return result

import time
import pickle
import os

folder='breaths_v7'

os.system(f'mkdir {folder}')
all_parameters = pickle.load(open('matched_paramters_w_triangle.p', 'rb'))

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='start')
    parser.add_argument('--finish', type=int, default=150, help='start')
    opts = parser.parse_args()
    return opts

args=get_args()

N_PROCESS=1
predictions = {}
parameters = {}
#exit()
total_runtime=time.time()
print(f"starting from breath {train['breath_id'].unique()[args.start]} to {train['breath_id'].unique()[args.finish-1]}")
for breath_nr, breath in enumerate(train['breath_id'].unique()[args.start:args.finish]):
    #if breath in done:
        # print(f'Skipping {breath}...')
        #continue

    start = time.time()
    if breath in all_parameters:
        n, preds, params = brute((breath_nr, breath), best_params=all_parameters[breath])
    else:
        n, preds, params = brute((breath_nr, breath))
    predictions[breath] = preds
    parameters[breath] = params
    # print(breath, n, params, time.time() - start)

    # pickle.dump(predictions, open(f'shujun_breaths/predictions_breath{breath}.p', 'wb+'))
    # pickle.dump(parameters, open(f'shujun_breaths/parameters_breath{breath}.p', 'wb+'))

    if breath_nr % 100 == 0 or breath_nr <= (args.finish - args.start - 2):
        pickle.dump(predictions, open(f'{folder}/predictions_breath{args.start}_{args.finish}.p', 'wb+'))
        pickle.dump(parameters, open(f'{folder}/parameters_breath{args.start}_{args.finish}.p', 'wb+'))
        #predictions={}
        #parameters={}
total_runtime=time.time()-total_runtime

print(f"total run time: {total_runtime}")

#first 10
#
"""
8 26 [4.0, 0.3, 10] 3.046066999435425
11 5 [-1, -1, -1] 3.0500271320343018
24 5 [-1, -1, -1] 3.0186567306518555
31 8 [8.0, 0.1, 20] 3.007828712463379
33 5 [-1, -1, -1] 3.0422236919403076
38 5 [-1, -1, -1] 3.0361177921295166
45 5 [-1, -1, -1] 3.022019386291504
50 6 [1.0, 0.5, 10] 3.025468587875366
51 5 [-1, -1, -1] 2.9775846004486084
"""
