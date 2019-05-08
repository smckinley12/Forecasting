'''
Useful functions for forecast analysis
Scott McKinley
'''
import xarray as xr
import pandas as pd
import numpy as np
from math import exp

'''
coarsen_data
inputs:
    data_ci: an xarray dataframe containing satellite data
returns:
    a dataframe containing values only at integer x,y
'''
def coarsen_data(data_ci):
    we_int = np.arange(240.0,280.0,dtype=np.float32)
    sn_int = np.arange(32.0,88.0,dtype=np.float32)
    data_crs = data_ci.sel(west_east=we_int,
                           south_north=sn_int)
    return data_crs


'''
build_datadf
inputs:
    data_files: a list of all satellite .nc files
    coordx, coordy: arrays of x,y values at which satellite data desired in df
returns:
    one pandas timeseries df with values for each coordinate
'''
def build_datadf(data_files, coordx, coordy):
    data_times = np.empty(0,dtype=np.datetime64)
    coord_ci = []
    for i in range(len(coordx)):
        coord_ci.append([])
    for data_file in data_files:
        data_ds = xr.open_dataset(data_file)
        data_times = np.concatenate([data_times,data_ds.time.values])
        data_ci = coarsen_data(data_ds.data_vars['ci'])
        for i in range(len(coordx)):
            x = coordx[i]
            y = coordy[i]
            ci = data_ci.sel(west_east=x,south_north=y).values
            coord_ci[i].extend(ci)
    coord_ci = np.array(coord_ci)
    data_df = pd.DataFrame({0:np.zeros(len(data_times))},index=data_times)
    for i in range(len(coordx)):
        data_df[i]=coord_ci[i]
    return data_df

'''
build_statdf
inputs:
    res_files: a list of ensemble-containing forecast results .nc files
            Note: coarsened ones require xr.open_dataarray.
    index: timeseries values, same as data_df timeseries
    columns: coordinate locations index for df columns, same as data_df columns
    coordx, coordy: arrays of x,y values to do statistics
    ci_min, ci_max: min/max average ci threshold value over field to 
            determine which frames are included in statistics.
returns:
    Eight pd dataframes, one avg, one std for four forecast times
'''
def build_statdf(res_files,index,columns,coordx,coordy,
                 ci_min = 0,ci_max = 1000):
    res_avg15_df = pd.DataFrame(index=index,columns=columns)
    res_std15_df = pd.DataFrame(index=index,columns=columns)
    res_avg30_df = pd.DataFrame(index=index,columns=columns)
    res_std30_df = pd.DataFrame(index=index,columns=columns)
    res_avg45_df = pd.DataFrame(index=index,columns=columns)
    res_std45_df = pd.DataFrame(index=index,columns=columns)
    res_avg60_df = pd.DataFrame(index=index,columns=columns)
    res_std60_df = pd.DataFrame(index=index,columns=columns)
    for res_file in res_files:
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        # Currently using avg over all times, ens members to determine
        # inclusion or exclusion from statistics, different later?
        ci_avg = np.average(res_ci.values)
        if(ci_avg<ci_min or ci_avg>ci_max):
            continue
        res_times = res_ci.time.values
        for c in range(len(coordx)):
            x = coordx[c]
            y = coordy[c]
            ci = res_ci.sel(west_east=x,south_north=y)
            for f in range(4):
                ci_avg = np.average(ci[f].values)
                ci_std = np.std(ci[f].values)
                try:
                    if(f==0):
                        res_avg15_df.at[res_times[f],c] = ci_avg
                        res_std15_df.at[res_times[f],c] = ci_std
                    elif(f==1):
                        res_avg30_df.at[res_times[f],c] = ci_avg
                        res_std30_df.at[res_times[f],c] = ci_std
                    elif(f==2):
                        res_avg45_df.at[res_times[f],c] = ci_avg
                        res_std45_df.at[res_times[f],c] = ci_std
                    elif(f==3):
                        res_avg60_df.at[res_times[f],c] = ci_avg
                        res_std60_df.at[res_times[f],c] = ci_std
                except:
                    continue
        res_ci.close()
    return (res_avg15_df,res_std15_df,res_avg30_df,res_std30_df,
            res_avg45_df,res_std45_df,res_avg60_df,res_std60_df)
    
'''
build_brier
inputs:
    res_files: a list of ensemble-containing forecast results .nc files
            Note: coarsened ones require xr.open_dataarray.
    index: timeseries values, same as data_df timeseries
    columns: coordinate locations index for df columns, same as data_df columns
    coordx, coordy: arrays of x,y values to do statistics
    thresh: binary threshold value for Brier Score calculation
returns:
    Four pd dataframes, one for each forecast time, containing timeseries
    values of the fractions of ensemble members at each coordinate that lie
    __BELOW__ the threshold value.
'''
def build_brier_fractions(res_files,index,columns,coordx,coordy,thresh):
    fr15 = pd.DataFrame(index=index,columns=columns)
    fr30 = pd.DataFrame(index=index,columns=columns)
    fr45 = pd.DataFrame(index=index,columns=columns)
    fr60 = pd.DataFrame(index=index,columns=columns)
    for res_file in res_files:
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        res_times = res_ci.time.values
        for c in range(len(coordx)):
            x = coordx[c]
            y = coordy[c]
            ci = res_ci.sel(west_east=x,south_north=y)
            for f in range(4):
                ens_vals = ci[f].values
                frac = float(len(np.where(ens_vals/thresh<1)[0]))/float(len(ens_vals))
                try:
                    if(f==0):
                        fr15.at[res_times[f],c] = frac
                    elif(f==1):
                        fr30.at[res_times[f],c] = frac
                    elif(f==2):
                        fr45.at[res_times[f],c] = frac
                    elif(f==3):
                        fr60.at[res_times[f],c] = frac
                except:
                    continue
        res_ci.close()
    return (fr15,fr30,fr45,fr60)

def build_weighted_fractions_ci(res_files,index,columns,coordx,coordy,thresh):
    print('weighted fractions')
    fr15 = pd.DataFrame(index=index,columns=columns)
    fr30 = pd.DataFrame(index=index,columns=columns)
    fr45 = pd.DataFrame(index=index,columns=columns)
    fr60 = pd.DataFrame(index=index,columns=columns)
    ### mask
    m = np.zeros((7,7))
    var = 2
    for i in range(7):
        for j in range(7):
            m[i,j] = exp(-((i-3)**2+(j-3)**2)/(2*var))        
    m = m/m[3,3]
    ###
    for res_file in res_files:
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        res_times = res_ci.time.values
        for c in range(len(coordx)):
            x = coordx[c]
            y = coordy[c]
            xwin = [x-3+i for i in range(7)]
            ywin = [y-3+i for i in range(7)]
            ci = res_ci.sel(west_east=xwin,south_north=ywin)
            for f in range(4):
                grid_vals = ci[f].values
                clear_count = 0
                for i in range(20):
                    weight_val = np.sum(np.multiply(m,grid_vals[i]))/np.sum(m)
                    if(weight_val/thresh<1):
                        clear_count += 1
                frac = clear_count/20
                #ones = np.zeros_like(grid_vals)
                #index = np.where(grid_vals/thresh<1)
                #ones[index]=1
                #grid_frac = np.sum(ones,axis=0)/20
                try:
                    if(f==0):
                        fr15.at[res_times[f],c] = frac
                    elif(f==1):
                        fr30.at[res_times[f],c] = frac
                    elif(f==2):
                        fr45.at[res_times[f],c] = frac
                    elif(f==3):
                        fr60.at[res_times[f],c] = frac
                except:
                    continue
        res_ci.close()
    return (fr15,fr30,fr45,fr60)  


def build_weighted_fractions_prob(res_files,index,columns,coordx,coordy,thresh):
    print('weighted fractions')
    fr15 = pd.DataFrame(index=index,columns=columns)
    fr30 = pd.DataFrame(index=index,columns=columns)
    fr45 = pd.DataFrame(index=index,columns=columns)
    fr60 = pd.DataFrame(index=index,columns=columns)
    ### mask
    m = np.zeros((7,7))
    var = 2
    for i in range(7):
        for j in range(7):
            m[i,j] = exp(-((i-3)**2+(j-3)**2)/(2*var))        
    m = m/m[3,3]
    ###
    for res_file in res_files:
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        res_times = res_ci.time.values
        for c in range(len(coordx)):
            x = coordx[c]
            y = coordy[c]
            xwin = [x-3+i for i in range(7)]
            ywin = [y-3+i for i in range(7)]
            ci = res_ci.sel(west_east=xwin,south_north=ywin)
            for f in range(4):
                grid_vals = ci[f].values
                #clear_count = 0
                #for i in range(20):
                #    weight_val = np.sum(np.multiply(m,grid_vals[i]))/np.sum(m)
                #    if(weight_val/thresh<1):
                #        clear_count += 1
                #frac = clear_count/20
                ones = np.zeros_like(grid_vals)
                index = np.where(grid_vals/thresh<1)
                ones[index]=1
                grid_frac = np.sum(ones,axis=0)/20
                frac = np.sum(np.multiply(m,grid_frac))/np.sum(m)
                try:
                    if(f==0):
                        fr15.at[res_times[f],c] = frac
                    elif(f==1):
                        fr30.at[res_times[f],c] = frac
                    elif(f==2):
                        fr45.at[res_times[f],c] = frac
                    elif(f==3):
                        fr60.at[res_times[f],c] = frac
                except:
                    continue
        res_ci.close()
    return (fr15,fr30,fr45,fr60) 



def build_benchmark_fractions(data_files,index,thresh):
    fr15 = pd.DataFrame(index=index)
    fr30 = pd.DataFrame(index=index)
    fr45 = pd.DataFrame(index=index)
    fr60 = pd.DataFrame(index=index)
    dt = np.timedelta64(900000000000,'ns') # 15 minute interval
    data_times = np.empty(0,dtype=np.datetime64)
    for data_file in data_files:
        print(data_file)
        data_ds = xr.open_dataset(data_file)
        data_times = data_ds.time.values
        data_ci = coarsen_data(data_ds.data_vars['ci'])
        ci = data_ci.values
        for i in range(len(data_times)):
            t = data_times[i]
            ci_flat = ci[i].flatten()
            wh = np.where(ci_flat/thresh<1)
            frac = len(ci_flat[wh])/len(ci_flat)
            for f in range(4):
                try:
                    if(f==0):
                        fr15.at[t+dt,0] = frac
                    elif(f==1):
                        fr30.at[t+2*dt,0] = frac
                    elif(f==2):
                        fr45.at[t+3*dt,0] = frac
                    elif(f==3):
                        fr60.at[t+4*dt,0] = frac
                except:
                    continue
        data_ds.close()
    return (fr15,fr30,fr45,fr60)
        

def build_solar_fractions_weightci(res_files,index,weights,columns,coordx,coordy,thresh):
    print('solar fractions')
    fr15 = pd.DataFrame(index=index,columns=[0])
    fr30 = pd.DataFrame(index=index,columns=[0])
    fr45 = pd.DataFrame(index=index,columns=[0])
    fr60 = pd.DataFrame(index=index,columns=[0])
    ci_store = np.zeros(len(coordx))
    ### mask, currently not used
    m = np.zeros((7,7))
    var = 2
    for i in range(7):
        for j in range(7):
            m[i,j] = exp(-((i-3)**2+(j-3)**2)/(2*var))        
    m = m/m[3,3]
    ###
    for res_file in res_files:
        print(res_file)
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        res_times = res_ci.time.values
        for f in range(4):
            ci_store = np.zeros((len(weights),20))
            for c in range(len(coordx)):
                x = coordx[c]
                y = coordy[c]
                w = weights[c]
                #xwin = [x-3+i for i in range(7)]
                #ywin = [y-3+i for i in range(7)]
                ci = res_ci.sel(west_east=x,south_north=y)
                ci_store[c] = w*ci[f].values
                #ones = np.zeros_like(grid_vals)
                #index = np.where(grid_vals/thresh<1)
                #ones[index]=1
                #grid_frac = np.sum(ones,axis=0)/20
            ens_vals = np.sum(ci_store,axis=0)/np.sum(weights)
            count_clear = 0
            for i in range(20):
                if(ens_vals[i]<thresh):
                    count_clear += 1
            frac = count_clear/20
            try:
                if(f==0):
                    fr15.at[res_times[f],0] = frac
                elif(f==1):
                    fr30.at[res_times[f],0] = frac
                elif(f==2):
                    fr45.at[res_times[f],0] = frac
                elif(f==3):
                    fr60.at[res_times[f],0] = frac
            except:
                continue
        res_ci.close()
    return (fr15,fr30,fr45,fr60)  
    
def build_solar_fractions_weightprob(res_files,index,weights,columns,coordx,coordy,thresh):
    print('solar fractions')
    fr15 = pd.DataFrame(index=index,columns=[0])
    fr30 = pd.DataFrame(index=index,columns=[0])
    fr45 = pd.DataFrame(index=index,columns=[0])
    fr60 = pd.DataFrame(index=index,columns=[0])
    ### mask, currently not used
    m = np.zeros((7,7))
    var = 2
    for i in range(7):
        for j in range(7):
            m[i,j] = exp(-((i-3)**2+(j-3)**2)/(2*var))        
    m = m/m[3,3]
    ###
    for res_file in res_files:
        print(res_file)
        res_ci = xr.open_dataarray(res_file)
        res_ci = res_ci.isel(time=slice(1,5))
        res_times = res_ci.time.values
        for f in range(4):
            prob_store = np.zeros(len(weights))
            for c in range(len(coordx)):
                x = coordx[c]
                y = coordy[c]
                w = weights[c]
                ci = res_ci.sel(west_east=x,south_north=y)
                ci_ens = ci[f].values
                prob_store[c] = w*len(ci_ens[np.where(ci_ens<thresh)])/20
            frac = np.sum(prob_store)/np.sum(weights)
            try:
                if(f==0):
                    fr15.at[res_times[f],0] = frac
                elif(f==1):
                    fr30.at[res_times[f],0] = frac
                elif(f==2):
                    fr45.at[res_times[f],0] = frac
                elif(f==3):
                    fr60.at[res_times[f],0] = frac
            except:
                continue
        res_ci.close()
    return (fr15,fr30,fr45,fr60) 




    