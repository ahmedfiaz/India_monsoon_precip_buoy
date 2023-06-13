import xarray as xr
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import xesmf as xe
import datetime as dt
from dateutil.relativedelta import relativedelta
from itertools import repeat
import thermodynamic_functions
import os
import xesmf as xe

class MaskIndianMonsoon:
    def __init__(self,land_mask,fil_cmz_mask):
        self.land_mask=land_mask
        self.lat=land_mask.lat
        self.lon=land_mask.lon
        self.wg_dict=dict(lat_min=7.5, lat_elb=18., lat_max=21.5, lon_min=72.25, lon_elb=74., lon_sw=76., lon_max=78.5)

        self.ghats_mask=xr.zeros_like(land_mask)
        self.sei_mask=xr.zeros_like(land_mask)
        cmz_mask=xr.open_mfdataset(fil_cmz_mask).dat
        self.cmz_mask=cmz_mask.where(cmz_mask!=-999)

        self.lat_str = "lat"
        self.lon_str = "lon"

        ### initialize masks ##
        self.mask_west_ghats()
        self.mask_sei()


    def __mask_upper_wg(self, lat, lon):
        wg=self.wg_dict
        return ((lat >= wg['lat_elb']) &
                (lat <= wg['lat_max']) &
                (lon >= wg['lon_min']) &
                (lon <= wg['lon_elb']))

    def __mask_lower_wg(self, lat, lon):
        wg=self.wg_dict
        cond1=lon <= wg['lon_max']\
              + (wg['lon_elb'] - wg['lon_max']) * (lat - wg['lat_min']) / (wg['lat_elb'] - wg['lat_min'])
        cond2=lon >= wg['lon_sw']\
              + (wg['lon_min'] - wg['lon_sw']) * (lat - wg['lat_min']) / (wg['lat_elb'] - wg['lat_min'])
        return ((lat >= wg['lat_min']) &
                (lat <= wg['lat_elb']) &
                cond1 & cond2)

    def __mask_wg(self, lat, lon):
        return (self.__mask_upper_wg(lat, lon) | self.__mask_lower_wg(lat, lon)).transpose("lat", "lon")

    def mask_west_ghats(self):
        """Mask the array outside of the Western Ghats region """
        wg_intermediate = self.__mask_wg(self.lat, self.lon)
        cond=(wg_intermediate == 1) & (self.land_mask == 1)
        self.ghats_mask=wg_intermediate.where(cond).drop(['sfc_area','land_mask','monsoon_mask'])

    def mask_sei(self):
        lat_max_se_india = 18.
        cond = (self.ghats_mask != 1) & (self.land_mask == 1)
        sei_mask=self.sei_mask.where(cond).where(self.lat <= lat_max_se_india).drop(['sfc_area','land_mask','monsoon_mask'])
        sei_mask+=1
        self.sei_mask=sei_mask

# class EnsoIndex:
#     def __init__(self,enso_file_path):
#         self.

class ProcessIMD(MaskIndianMonsoon):
    def __init__(self,file_path,land_mask,fil_cmz_mask):
        self.ds_rain=xr.open_mfdataset(file_path)
        super().__init__(land_mask,fil_cmz_mask)


init_dict=lambda x: dict.fromkeys(['wg', 'cmz', 'sei'])

class ProcessERA5(MaskIndianMonsoon):

    def __init__(self,era5_file_path, dir_out, filter_surf_press,land_mask,fil_cmz_mask, save_dir):

        self.era5_file=era5_file_path
        self.date_string=era5_file_path.split('/')[-1].split('Tq_')[1].split('.grib')[0]
        self.dir_out=dir_out
        self.filter_surf_press=filter_surf_press
        self.save_dir=save_dir

        path=era5_file_path.split('era-5')[0]+era5_file_path.split('/')[-1].split('Tq_')[0]
        self.filsrf=self.modify_original_path_get_new_path(path,'surf',self.date_string)
        self.era5_processing=True

        ## initialize empty thermo vars and dicts
        self.qmain=self.Tmain=self.d2m_main=self.t2m_main=self.sp_main=None
        self.q,self.T,self.d2m,self.t2m,self.sp,self.pbl_top=list(map(init_dict,repeat(None,6)))
        self.qsat,self.thetae_sat,self.thetae=list(map(init_dict,repeat(None,3)))
        self.thetae_2m,self.thetae_bl,self.thetae_lft,self.thetae_sat_lft=list(map(init_dict,repeat(None,4)))
        self.pbl_top_lev=init_dict(None)
        self.lev=None

        super().__init__(land_mask,fil_cmz_mask)
        self.mask=dict(wg=self.ghats_mask, cmz=self.cmz_mask,
                       sei=self.sei_mask)


    @staticmethod
    def modify_original_path_get_new_path(path:str,var:str,date_string:str)->'new_path':
        return path+var+'_'+date_string+'.grib'

    mask_flatten_drop=lambda self,x,mask:(x*mask).stack(z=('lat','lon')).dropna('z')

    @staticmethod
    def __filter_surf_press(var_list, cond):
        return [i.where(cond).dropna('z') for i in var_list]

    @staticmethod
    def __get_era5_vars(fil, var_list, surf=False):

        ds = xr.open_mfdataset(fil)
        if surf:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"}).drop(['step', 'number', 'valid_time'])
        else:
            ## for vertical levels, only select upto 500 hPa ##
            ds = ds.rename({"latitude": "lat", "longitude": "lon",
                            'isobaricInhPa': 'lev'}).drop(['step', 'number', 'valid_time'])
            ds = ds.sel(lev=slice(1000, 150))

        var_list = [ds[i] for i in var_list]
        ds.close()
        return var_list

    def extract_era5(self):

        # print('     get vars')

        ### read era5  & q ###
        self.qmain, self.Tmain = self.__get_era5_vars(self.era5_file, ['q', 't'])
        ### read era5 surf. ###
        self.d2m_main, self.t2m_main, self.sp_main = self.__get_era5_vars(self.filsrf, ['d2m', 't2m', 'sp'], surf=True)  ## dewpoint temp
        self.lev = self.Tmain.lev
        self.sp_main*= 1e-2  ## convert surf. pressure to hPa

    def flatten_mask_era5(self, key):

        ## loop through masks ##
        mask=self.mask[key]

        qmasked, Tmasked = list(map(self.mask_flatten_drop, [self.qmain, self.Tmain], repeat(mask)))
        d2m_masked, t2m_masked, sp_masked = list(map(self.mask_flatten_drop,
                                                     [self.d2m_main, self.t2m_main, self.sp_main], repeat(mask)))
        cond_filter = sp_masked > self.filter_surf_press  ### surf. pressure filter

        print('     mask flatten')
        self.q[key], self.T[key] = self.__filter_surf_press([qmasked, Tmasked], cond_filter)
        self.d2m[key], self.t2m[key]= self.__filter_surf_press([d2m_masked, t2m_masked], cond_filter)
        self.sp[key] = self.__filter_surf_press([sp_masked], cond_filter)[0]
        self.pbl_top[key] = self.sp[key] - 1e2  ## a 100-hpa-thick layer above surface

    ### thermo computation block ###
    def __compute_surface_thetae(self,key):
        sp = self.sp[key]
        t2m = self.t2m[key]
        d2m = self.d2m[key]
        q2m = thermodynamic_functions.qs_calc(sp, d2m)  # saturation sp. humidity at dew point temperature
        self.thetae_2m[key]=thermodynamic_functions.theta_e_calc(sp,t2m,q2m)

    def __compute_qsat(self,key):
        temp=self.T[key]
        self.qsat[key]=thermodynamic_functions.qs_calc(self.lev,temp)

    def __compute_thetae_thetae_sat(self,key):
        temp = self.T[key]
        q = self.q[key]
        qsat = self.qsat[key]
        self.thetae[key] = thermodynamic_functions.theta_e_calc(self.lev,temp,q)
        self.thetae_sat[key] = thermodynamic_functions.theta_e_calc(self.lev,temp,qsat)

    def compute_thermo(self,key):
        self.__compute_surface_thetae(key)
        self.__compute_qsat(key)
        self.__compute_thetae_thetae_sat(key)

    ### end thermo computation block ###

    @staticmethod
    def __perform_layer_ave(var,lev):

        dp=abs(lev.diff('lev')).assign_coords({"lev": np.arange(0,lev.size-1)})
        var1=var.isel(lev=slice(0,lev.size-1)).assign_coords({"lev": np.arange(0,lev.size-1)})
        var2=var.isel(lev=slice(1,lev.size)).assign_coords({"lev": np.arange(0,lev.size-1)})
        return ((var1+var2)*dp*0.5).sum('lev')

    @staticmethod
    def get_surf_contribution_bl_ave(var, var_surf, sp, lev):

        sp_diff = sp - lev
        sp_diff = sp_diff.where(sp_diff >= 0)
        var_surf_nearest = var.isel(lev=sp_diff.argmin('lev'))
        sp_diff = sp_diff.isel(lev=sp_diff.argmin('lev'))

        return (var_surf_nearest + var_surf) * 0.5 * sp_diff

    def __bl_ave(self, var, var_surf, key):

        lev = self.lev
        sp = self.sp[key]
        pbl_top = self.pbl_top[key]

        ### get trop. contribution ###
        cond = np.logical_and(lev >= pbl_top, lev <= sp)
        var_bl = var.where(cond)
        self.pbl_top_lev[key] = xr.where(np.isfinite(var_bl), lev, np.nan).idxmin('lev')
        pbl_thickness = sp - self.pbl_top_lev[key]
        var_bl_contribution = self.__perform_layer_ave(var_bl, lev)

        ### get surface contribution ###
        near_surf_idx = xr.where(np.isfinite(var_bl), lev, np.nan).argmax('lev').compute()
        var_near_surf = var_bl.isel(lev=near_surf_idx)
        sp_diff = (sp - lev).where(sp >= lev).compute()
        sp_diff = sp_diff.isel(lev=sp_diff.argmin('lev'))
        var_surf_contribution = (var_near_surf + var_surf) * 0.5 * sp_diff

        return (var_bl_contribution + var_surf_contribution) / pbl_thickness

    def __lft_ave(self, var, key):

        pbl_top = self.pbl_top[key]
        lev = self.lev
        pbl_top_lev = self.pbl_top_lev[key]

        var_lft = var.where(lev <= pbl_top)
        lft_top_lev = xr.where(np.isfinite(var_lft), lev, np.nan).idxmin('lev')
        lft_thickness = pbl_top_lev - lft_top_lev
        var_lft_contribution = self.__perform_layer_ave(var_lft, lev)

        return var_lft_contribution / lft_thickness

    def get_layer_averaged_thermo(self,key):

        thetae = self.thetae[key]
        thetae_sat = self.thetae_sat[key]
        thetae_2m = self.thetae_2m[key]

        self.thetae_bl[key] = self.__bl_ave(thetae, thetae_2m,key)
        self.thetae_lft[key] = self.__lft_ave(thetae,key)
        self.thetae_sat_lft[key] = self.__lft_ave(thetae_sat,key)

    def save_netcdf(self):
        file_path=self.save_dir+'layer_avg_thetae_{}.nc'.format(self.date_string)
        ds = {}
        for key in self.mask.keys():
            ds[key] = xr.Dataset(data_vars=dict(thetae_bl=self.thetae_bl[key].unstack(),
                                                thetae_lft=self.thetae_lft[key].unstack(),
                                                thetae_sat_lft=self.thetae_sat_lft[key].unstack()))\
                .drop(['surface', 'lev'])

        ds_new = xr.merge([ds[key] for key in ds.keys()], join='outer')
        save_netcdf_with_check(ds_new,file_path) ### using global method here

    def main(self):
        self.extract_era5()
        for key in self.mask.keys():
            print(key)
            self.flatten_mask_era5(key)
            self.compute_thermo(key)
            self.get_layer_averaged_thermo(key)

        self.save_netcdf()


    # def check_era5_vars(self):
    #
    #     ## save files ###
    #     if self.SAVE_REGION == 'OCN':
    #         region_str1 = 'ocn'
    #         region_str2 = 'oceans'
    #
    #     elif self.SAVE_REGION == 'LND':
    #         region_str1 = 'lnd'
    #         region_str2 = 'land'
    #
    #     self.HBL_FILE = self.dir_out \
    #                     + '/{}/hbl_{}/'.format(region_str1, region_str1) + "hbl_{}_{}.npy".format(region_str2,
    #                                                                                               self.date_string)
    #
    #     self.HLFT_FILE = self.dir_out \
    #                      + '/{}/hlft_{}/'.format(region_str1, region_str1) + "hlft_{}_{}.npy".format(region_str2,
    #                                                                                                  self.date_string)
    #
    #     self.HSAT_LFT_FILE = self.dir_out \
    #                          + '/{}/hsat_lft_{}/'.format(region_str1, region_str1) + "hsat_lft_{}_{}.npy".format(
    #         region_str2, self.date_string)
    #
    #     cond1 = os.path.isfile(self.HBL_FILE)
    #     cond2 = os.path.isfile(self.HLFT_FILE)
    #     cond3 = os.path.isfile(self.HSAT_LFT_FILE)
    #
    #     if (not cond1 or not cond2) or (not cond3):
    #         self.era5_processing = True
    #     else:
    #         self.era5_processing = False



### A few convenience functions ###

def save_netcdf_with_check(ds,save_file):
    if os.path.isfile(save_file):
        os.remove(save_file)
    ds.to_netcdf(save_file)
    print('File saved as {}'.format(save_file))


def convert_nino34_indx_to_netcdf(nino34_file,save_file):
    nino34_array = np.loadtxt(nino34_file, skiprows=1, max_rows=153)
    date_list=[]
    nino34_indx=[]
    for i in nino34_array:
        date_list.append([dt.datetime(i[0].astype('int'),j+1,1) for j in range(i[1:].size)])
        nino34_indx.append([j for j in i[1:]])

    date_list=[j for i in date_list for j in i]
    nino34_indx=[j for i in nino34_indx for j in i]
    ds_nino34=xr.Dataset(data_vars=dict(nino34=(['time'],nino34_indx)),
              coords=dict(time=date_list))
    ## apply 3-month running mean
    ds_nino34=ds_nino34.rolling(time=3, center=True).mean().dropna("time")
    save_netcdf_with_check(ds_nino34,save_file)
    ds_nino34.close()


def regrid_cmz_mask(cmz_mask, ds_out, save_file):

    regridder = xe.Regridder(cmz_mask, ds_out, method='nearest_s2d', reuse_weights=False, periodic=False)
    cmz_mask_regridded = regridder(cmz_mask).drop(['sfc_area','land_mask','monsoon_mask'])
    save_netcdf_with_check(cmz_mask_regridded, save_file)
    cmz_mask_regridded.close()
