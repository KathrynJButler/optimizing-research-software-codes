import datetime
import multiprocessing
import os
import re
import sys
import math

from typing import List

import compute
import config

import yaml
import numpy as np
import pandas as pd
import georinex as gr

from timer_func import timeit, print_timing_stats #TIME CHECKER!!!

class Rinex(object):
  _leap_months = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
  _normal_months = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
  _sec_per_day = 86400
  _sec_per_week = 604800
  _sec_half_week = 302400
  _gpst_0 = 3822681600

  class SatPos(object):
    """
    Satellite Position Class.
    Stores location (x, y, z), velocity (vx, vy, vz) and acceleration (ax, ay, az)
    """
    @timeit #TIME CHECKER!!!
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, ax=0, ay=0, az=0):
      """
      Initialize SatPos object. Default value 0
      """
      self.x = x
      self.y = y
      self.z = z
      self.vx = vx
      self.vy = vy
      self.vz = vz
      self.ax = ax
      self.ay = ay
      self.az = az

    @classmethod
    @timeit #TIME CHECKER!!!
    def copy(cls, other):
      """
      Copy SatPos object
      :param other: object to copy
      :return: copied SatPos object
      """
      return cls(other.x, other.y, other.z, other.vx, other.vy, other.vz, other.ax, other.ay, other.az)

  @timeit #TIME CHECKER!!!
  def __init__(self, nav_files: List[str], obs_files: List[str], constellations: List[str]=None, output_dir: str=None, year: int=None, doy: int=None, hour_minute_index: str=None):
    """
    Initialize Rinex object
    :param nav_file: navigation rinex file
    :param obs_file: observation rinex file
    """
    if constellations is None:
      constellations = ['G', 'R', 'E', 'C']
    self.constellations = constellations
    self.nav_files = nav_files
    self.obs_files = obs_files
    self.output_dir = output_dir
    self.year = year
    self.doy = doy
    self.hour_minute_index = hour_minute_index

    self.config = config.load_config()  # Load configuration
    self.config['etc']['leap_second'] = sorted(self.config['etc']['leap_second'])  # Sort leap_second value

    tmp = []
    for nav in self.nav_files:
      tmp.append(self._preprocess_nav(nav))
    self.nav = pd.concat(tmp)
    tmp = []
    for obs in self.obs_files:
      self._preprocess_obs(obs)
      tmp.append(gr.load(obs).to_dataframe())
    self.obs = pd.concat(tmp)

    # Call parse
    #self.parse()

  @timeit #TIME CHECKER!!!
  def _preprocess_nav(self, nav):
    try:
      return gr.load(nav).to_dataframe()
    except:
      return self._nav_to_version_2(nav)

  @timeit #TIME CHECKER!!!
  def _preprocess_obs(self, obs):
    file_stat = os.stat(obs)
    if self.config['rinex'].get('obs_size_threashold', 64) < (file_stat.st_size / 1024 / 1024):
      if self._get_obs_version(obs) == 3:
        print('large obs version 3 file detected')
        self._obs_to_version_2(obs)

  @timeit #TIME CHECKER!!!
  def _get_obs_version(self, obs):
    with open(obs, 'r') as fp:
      first_line = fp.readline()
      version_string = first_line.strip().split(' ')[0]
      return int(version_string.split('.')[0]) # return major version

  @timeit #TIME CHECKER!!!
  def _obs_to_version_2(self, obs):
    print('obs version down to 2')
    import subprocess
    executable = {
      'linux': 'gfzrnx_2.1.9_lx64',
      'win32': 'gfzrnx_1.16-8204_win64.exe'
    }.get(sys.platform)
    executable = os.path.join('gfzrnx', executable)
    try:
      os.chmod(executable, 0o755)
    except:
      pass
    subprocess.Popen([executable, '-finp', obs, '-fout', obs, '-f', '--version_out', '2', '-satsys', ''.join(self.constellations)]).wait()

  @timeit #TIME CHECKER!!!
  def _nav_to_version_2(self, nav):
    print('nav version down to 2')
    import subprocess
    executable = {
      'linux': 'gfzrnx_2.1.9_lx64',
      'win32': 'gfzrnx_1.16-8204_win64.exe'
    }.get(sys.platform)
    executable = os.path.join('gfzrnx', executable)
    try:
      os.chmod(executable, 0o755)
    except:
      pass
    tmp = []
    for c in self.config['constellations']:
      subprocess.Popen([executable, '-finp', nav, '-fout', nav + c, '-f', '--version_out', '2', '-satsys', c, '-errlog', f'{nav}_{c}_conv_log.txt']).wait()
      try:
        tmp.append(gr.load(nav + c).to_dataframe())
      except:
        pass # skip if issue still occurs
    return pd.concat(tmp)
    
  @timeit #TIME CHECKER!!!
  def parse(self) -> pd.DataFrame:
    """
    Parse observation and navigation file and calculate position, azimuth, and elevation
    """

    # Filter
    #self.obs = self.obs[
    #  (self.obs.C1.notna()) | (self.obs.C2.notna()) | (self.obs.C5.notna()) | (self.obs.C7.notna()) | (
    #    self.obs.C8.notna())]
    tmp = []
    for c in self.constellations:
      tmp.append(self.obs[self.obs.index.get_level_values('sv').str.startswith(c)])
    self.obs = pd.concat(tmp)
    self.nav = self.nav[(self.nav.SVclockBias.notna())]

    # Add additional fields
    for field in ['PosX', 'PosY', 'PosZ', 'ClkCorr', 'FreqBand', 'SignalStrength', 'CarrierPhase', 'Doppler', 'CA', 'P', 'MJS']:
      self.obs.insert(self.obs.shape[1], field, np.nan)
    self.nav.insert(self.nav.shape[1], 'MJS', -1)

    # Add MJS Timestamp
    try:
      self.obs['MJS'] = self.obs.apply(lambda x: self.to_mjd(x.name[1]), axis=1)
    except:
      self.obs['MJS'] = self.obs.apply(lambda x: self.to_mjd(x.name[0]), axis=1)
    try:
      self.nav['MJS'] = self.nav.apply(lambda x: self.to_mjd(x.name[1], x.name[0]), axis=1)
    except:
      self.nav['MJS'] = self.nav.apply(lambda x: self.to_mjd(x.name[0], x.name[1]), axis=1)

    # Calculate satellite positions
    self.calculate_positions()

    # filter out empty x, y, z
    self.obs = self.obs[(self.obs.PosX.notna()) & (self.obs.PosY.notna()) & (self.obs.PosY.notna())]

    # sort by time,satid and save to csv
    #self.obs.sort_values(['time', 'sv'])[['MJS', 'PosX', 'PosY', 'PosZ']].to_csv(os.path.join('data', 'compactdb.csv'))

    # write parsed data to csv
    if self.output_dir is not None:
      output_file = os.path.join(self.output_dir, self.config['station']['id'], 'compactdb', f'{self.year}-{self.doy}')
      if self.hour_minute_index is not None:
        output_file += f'-{self.hour_minute_index}'
      output_file += '.csv'
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
    else:
      output_file = os.path.join(self.config['rinex']['output_dir'], self.config['station']['id'],
                                 'compactdb', f'{self.year}-{self.doy}')
      if self.hour_minute_index is not None:
        output_file += f'-{self.hour_minute_index}'
      output_file += '.csv'
      os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fp = open(output_file, 'w')
    try:
      # write csv header
      fp.write(
        'SatID,Time,MJS,ClkCorr,PosX,PosY,PosZ,AziAngle,ElevAngle,FreqBand,SignalStrength,CarrierPhase,Doppler,CA,P\n')
      self.obs = self.obs.sort_values(['time', 'sv'])
      # iterate items
      for item in self.obs.itertuples():
        res = self.get_additional_fields(item)
        elevation, azimuth = compute.calculate_elevation_azimuth(np.array(self.config['station']['info']),
                                                                 np.array([item.PosX, item.PosY, item.PosZ]))
        if isinstance(item.Index[0], str):
          prn_index = 0
          dt_index = 1
        else:
          prn_index = 1
          dt_index = 0
        for r in res:
          # Filter
          #r[3] = '' if np.isnan(r[3]) or int(r[3]) < 0 else r[3]  # ignore doppler if it is NaN or negative
          #r[4] = '' if np.isnan(r[4]) else r[4]  # ignore CA if it is NaN (to prevent nan string in csv file)
          fp.write(
            f'{item.Index[prn_index]},{item.Index[dt_index]},{item.MJS},{item.ClkCorr},{item.PosX},{item.PosY},{item.PosZ},{azimuth},{elevation},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n')
    except:
      import traceback
      traceback.print_exc()
    finally:
      fp.close()
    return pd.read_csv(output_file)

  @timeit #TIME CHECKER!!!
  def calculate_positions(self):
    try:
      if self.config['etc'].get('disable_parallel', False):
        raise Exception('Throw exception to disable parallel')
      num_procs = self.config['etc'].get('num_cores', 0)
      if num_procs == 0:
        num_procs = multiprocessing.cpu_count() // 2
      print(f'Using {num_procs} cores to calculate positions.\nIf you encounter memory errors, reduce num_cores and try it again or disable parallel')
      obs_partition = np.array_split(self.obs, num_procs)
      res = None
      with multiprocessing.Pool(processes=num_procs) as pool:
        res = pool.map(self.get_satellite_position_process_worker, obs_partition)
      self.obs = pd.concat(res)
    except:
      import traceback
      traceback.print_exc()
      print('Unable to process parallel. Use single core instead')
      self.obs[['PosX', 'PosY', 'PosZ', 'ClkCorr']] = self.obs.apply(lambda x: self.get_satellite_position(x), axis=1,
                                                                     result_type='expand')

  @timeit #TIME CHECKER!!!
  def get_satellite_position_process_worker(self, sat_list):
    sat_list[['PosX', 'PosY', 'PosZ', 'ClkCorr']] = sat_list.apply(lambda x: self.get_satellite_position(x), axis=1,
                                                                   result_type='expand')
    return sat_list

  @timeit #TIME CHECKER!!!
  def get_additional_fields(self, sat) -> list:
    """
    return additional fields (freq_band, Signal Strength, Carrier Phase, Doppler, CA, P)
    :param sat: Satellite Data (Pandas Series)
    :return: list of additional fields
    [
      [freq_band, Signal Strength, Carrier Phase, Doppler, CA, P]
      ...
    ]
    """
    if isinstance(sat.Index[0], str):
      sat_id = sat.Index[0]
    else:
      sat_id = sat.Index[1]
    bands = []
    if sat_id.startswith('G'):   # GPS
      if self.has_freq_band(sat, ['L1', 'C1', 'P1', 'D1', 'S1']):
        bands.append([0, getattr(sat, 'S1', ''), getattr(sat, 'L1', ''), getattr(sat, 'D1', ''), getattr(sat, 'C1', ''), getattr(sat, 'P1', '')])
      if self.has_freq_band(sat, ['L2', 'C2', 'P2', 'D2', 'S2']):
        bands.append([1, getattr(sat, 'S2', ''), getattr(sat, 'L2', ''), getattr(sat, 'D2', ''), getattr(sat, 'C2', ''), getattr(sat, 'P2', '')])
      if self.has_freq_band(sat, ['L5', 'C5', 'D5', 'S5']):
        bands.append([2, getattr(sat, 'S5', ''), getattr(sat, 'L5', ''), getattr(sat, 'D5', ''), getattr(sat, 'C5', ''), getattr(sat, 'P5', '')])

      if len(bands) == 0:
        # Rinex3 names
        if self.has_freq_band(sat, ['L1C', 'C1C', 'C1W', 'S1C']):
          bands.append([0, getattr(sat, 'S1C', ''), getattr(sat, 'L1C', ''), getattr(sat, 'D1', ''), getattr(sat, 'C1C', ''), getattr(sat, 'C1W', '')])
        if self.has_freq_band(sat, ['L2W', 'C2L', 'C2W', 'S2W']):
          bands.append([1, getattr(sat, 'S2W', ''), getattr(sat, 'L2W', ''), getattr(sat, 'D2', ''), getattr(sat, 'C2L', ''), getattr(sat, 'C2W', '')])
        if self.has_freq_band(sat, ['L5Q', 'C5Q', 'S5Q']):
          bands.append([2, getattr(sat, 'S5Q', ''), getattr(sat, 'L5Q', ''), getattr(sat, 'D5', ''), getattr(sat, 'C5Q', ''), getattr(sat, 'P5', '')])
      return bands
    elif sat_id.startswith('R'):  # GLONASS
      if self.has_freq_band(sat, ['L1', 'C1', 'P1', 'D1', 'S1']):
        bands.append([3, getattr(sat, 'S1', ''), getattr(sat, 'L1', ''), getattr(sat, 'D1', ''), getattr(sat, 'C1', ''), getattr(sat, 'P1', '')])
      if self.has_freq_band(sat, ['L2', 'C2', 'P2', 'D2', 'S2']):
        bands.append([4, getattr(sat, 'S2', ''), getattr(sat, 'L2', ''), getattr(sat, 'D2', ''), getattr(sat, 'C2', ''), getattr(sat, 'P2', '')])
      if self.has_freq_band(sat, ['L7', 'C7', 'D7', 'S7']):
        bands.append([5, getattr(sat, 'S7', ''), getattr(sat, 'L7', ''), getattr(sat, 'D7', ''), getattr(sat, 'C2', ''), ''])

      if len(bands) == 0:
        # for rinex ver_3
        band = self.get_freq_band_rinex3(3, sat, ['S1', 'L1', 'D1', 'C1', 'P1'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(4, sat, ['S2', 'L2', 'D2', 'C2', 'P2'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(5, sat, ['S7', 'L7', 'D7', 'C7', 'P7'])
        if band is not None:
          bands.append(band)
      return bands
    elif sat_id.startswith('E'):  # Galileo
      if self.has_freq_band(sat, ['L1', 'C1', 'P1', 'D1', 'S1']):
        bands.append([6, getattr(sat, 'S1', ''), getattr(sat, 'L1', ''), getattr(sat, 'D1', ''), getattr(sat, 'C1', ''), getattr(sat, 'P1', '')])
      if self.has_freq_band(sat, ['L5', 'C5', 'D5', 'S5']):
        bands.append([7, getattr(sat, 'S5', ''), getattr(sat, 'L5', ''), getattr(sat, 'D5', ''), getattr(sat, 'C5', ''), ''])
      if self.has_freq_band(sat, ['L6', 'C6', 'D6', 'S6']):
        bands.append([10, getattr(sat, 'S6', ''), getattr(sat, 'L6', ''), getattr(sat, 'D6', ''), getattr(sat, 'C6', ''), ''])
      if self.has_freq_band(sat, ['L7', 'C7', 'D7', 'S7']):
        bands.append([8, getattr(sat, 'S7', ''), getattr(sat, 'L7', ''), getattr(sat, 'D7', ''), getattr(sat, 'C7', ''), ''])
      if self.has_freq_band(sat, ['L8', 'C8', 'D8', 'S8']):
        bands.append([9, getattr(sat, 'S8', ''), getattr(sat, 'L8', ''), getattr(sat, 'D8', ''), getattr(sat, 'C8', ''), ''])

      if len(bands) == 0:
        # for rinex ver_3
        band = self.get_freq_band_rinex3(6, sat, ['S1', 'L1', 'D1', 'C1', 'P1'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(7, sat, ['S5', 'L5', 'D5', 'C5', 'P5'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(10, sat, ['S6', 'L6', 'D6', 'C6', 'P6'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(8, sat, ['S7', 'L7', 'D7', 'C7', 'P7'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(9, sat, ['S8', 'L8', 'D8', 'C8', 'P8'])
        if band is not None:
          bands.append(band)
      return bands
    elif sat_id.startswith('C'):  # Beidou
      if self.has_freq_band(sat, ['L1', 'C1', 'P1', 'D1', 'S1']):
        bands.append([11, getattr(sat, 'S1', ''), getattr(sat, 'L1', ''), getattr(sat, 'D1', ''), getattr(sat, 'C1', ''), ''])
      if self.has_freq_band(sat, ['L2', 'C2', 'P2', 'D2', 'S2']):
        bands.append([11, getattr(sat, 'S2', ''), getattr(sat, 'L2', ''), getattr(sat, 'D2', ''), getattr(sat, 'C2', ''), ''])
      if self.has_freq_band(sat, ['L5', 'C5', 'D5', 'S5']):
        bands.append([13, getattr(sat, 'S5', ''), getattr(sat, 'L5', ''), getattr(sat, 'D5', ''), getattr(sat, 'C5', ''), ''])
      if self.has_freq_band(sat, ['L6', 'C6', 'D6', 'S6']):
        bands.append([14, getattr(sat, 'S6', ''), getattr(sat, 'L6', ''), getattr(sat, 'D6', ''), getattr(sat, 'C6', ''), ''])
      if self.has_freq_band(sat, ['L7', 'C7', 'D7', 'S7']):
        bands.append([12, getattr(sat, 'S7', ''), getattr(sat, 'L7', ''), getattr(sat, 'D7', ''), getattr(sat, 'C7', ''), ''])
      if self.has_freq_band(sat, ['L8', 'C8', 'D8', 'S8']):
        bands.append([16, getattr(sat, 'S8', ''), getattr(sat, 'L8', ''), getattr(sat, 'D8', ''), getattr(sat, 'C8', ''), ''])

      if len(bands) == 0:
        # for rinex ver_3
        band = self.get_freq_band_rinex3(11, sat, ['S2', 'L2', 'D2', 'C2', 'P2'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(13, sat, ['S5', 'L5', 'D5', 'C5', 'P5'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(14, sat, ['S6', 'L6', 'D6', 'C6', 'P6'])
        if band is not None:
          bands.append(band)
        band = self.get_freq_band_rinex3(16, sat, ['S8', 'L8', 'D8', 'C8', 'P8'])
        if band is not None:
          bands.append(band)
      return bands
  @timeit #TIME CHECKER!!!
  def has_freq_band(self, sat, prop=[]):
    for p in prop:
      if not np.isnan(getattr(sat, p, np.nan)):
        return True
    return False

  @timeit #TIME CHECKER!!!
  def get_first_field(self, sat, prop):
    for p in prop:
      if not np.isnan(getattr(sat, p, np.nan)):
        return getattr(sat, p)
    return 'None'

  @timeit #TIME CHECKER!!!
  def get_freq_band_rinex3(self, band, sat, prop):
    attrs = dir(sat)
    res = [band]
    valid = False
    for p in prop:
      sat_prop = list(filter(lambda k: k.startswith(p), attrs))
      local_res = ''
      if len(sat_prop) > 0:
        for c in sat_prop:
          if not np.isnan(getattr(sat, c, np.nan)):
            local_res = getattr(sat, c)
            valid = True
            break
        res.append(local_res)
      else:
        res.append('')
    if valid:
      return res
    return None

  @timeit #TIME CHECKER!!!
  def get_first_c(self, el):
    if pd.notna(el.get('C1')):
      return el.C1
    if pd.notna(el.get('C1C')):
      return el.C1C
    if pd.notna(el.get('C1P')):
      return el.C1P
    elif pd.notna(el.get('C2')):
      return el.C2
    elif pd.notna(el.get('C2L')):
      return el.C2L
    elif pd.notna(el.get('C2I')):
      return el.C2I
    elif pd.notna(el.get('C5')):
      return el.C5
    elif pd.notna(el.get('C5P')):
      return el.C5P
    elif pd.notna(el.get('C5Q')):
      return el.C5Q
    elif pd.notna(el.get('C6')):
      return el.C6
    elif pd.notna(el.get('C6I')):
      return el.C6I
    elif pd.notna(el.get('C7')):
      return el.C7
    elif pd.notna(el.get('C7I')):
      return el.C7I
    elif pd.notna(el.get('C7Q')):
      return el.C7Q
    elif pd.notna(el.get('C8')):
      return el.C8
    return -1

  @timeit #TIME CHECKER!!!
  def get_first_p(self, el):
    if 'P2' not in el and 'C2W' not in el:
      return -1
    if pd.notna(el.get('P2')):
      return el.P2
    elif pd.notna(el.get('C2W')):
      return el.C2W
    else:
      return -1

  @timeit #TIME CHECKER!!!
  def get_satellite_position_partition(self, df):
    for i in df.itertuples():
      df[['PosX', 'PosY', 'PosZ', 'ClkCorr']] = self.get_satellite_position(i)
    return df

  @timeit #TIME CHECKER!!!
  def get_satellite_position(self, sat):
    try:
      if isinstance(sat.name[0], str):
        prn = sat.name[0]
      else:
        prn = sat.name[1]
      obs_codes = [self.get_first_c(sat), self.get_first_p(sat)]
      if -1 in obs_codes:
        obs_codes.remove(-1)
      observation_mean = sum(obs_codes) / len(obs_codes)
      if prn.startswith('R'):
        return self.get_satellite_position_glonass(prn, sat.MJS, observation_mean)
      else:
        return self.get_satellite_position_std(prn, sat.MJS, observation_mean)
    except:
      return None, None, None, None

  @timeit #TIME CHECKER!!!
  def get_satellite_position_glonass(self, prn, t, observation):
    j2 = -1.0826257e-3
    omegae_dot = 7.292115e-5
    gm = 3.986004418e14

    ell_a = 6378136

    blk = self.find_nearest_block(prn, t)
    if blk is None:
      return None, None, None, None

    p0 = Rinex.SatPos(blk.X, blk.Y, blk.Z, blk.dX, blk.dY, blk.dZ, blk.dX2, blk.dY2, blk.dZ2)

    taun = blk.SVclockBias
    gamman = blk.SVrelFreqBias

    time_tx_raw = t - observation / 299792458.0
    clkcorr = taun - gamman * (time_tx_raw - blk.MJS)
    time_tx = time_tx_raw - clkcorr
    dtc = time_tx - blk.MJS
    satoff = -taun + gamman * dtc
    tk = time_tx - blk.MJS - satoff

    int_step = 60
    n = math.floor(math.fabs(tk / int_step))
    int_step_res = math.fmod(tk, int_step)
    ii = []
    if tk < 0:
      ii = [-int_step for x in range(n)]
    else:
      ii = [int_step for x in range(n)]

    if int_step_res != 0:
      n += 1
      ii.append(int_step_res)

    pos = Rinex.SatPos.copy(p0)
    for step in ii:
      p1 = None
      p2 = Rinex.SatPos()
      p3 = Rinex.SatPos()
      p4 = Rinex.SatPos()
      d1 = None
      d2 = Rinex.SatPos()
      d3 = Rinex.SatPos()
      d4 = Rinex.SatPos()

      p1 = Rinex.SatPos.copy(pos)
      d1 = self.glonass_diffeq(ell_a, gm, j2, omegae_dot, p0, p1)

      p2.x = pos.x + d1.x * step / 2.0
      p2.y = pos.y + d1.y * step / 2.0
      p2.z = pos.z + d1.z * step / 2.0
      p2.vx = pos.vx + d1.vx * step / 2.0
      p2.vy = pos.vy + d1.vy * step / 2.0
      p2.vz = pos.vz + d1.vz * step / 2.0

      d2 = self.glonass_diffeq(ell_a, gm, j2, omegae_dot, p0, p2)

      p3.x = pos.x + d2.x * step / 2.0
      p3.y = pos.y + d2.y * step / 2.0
      p3.z = pos.z + d2.z * step / 2.0
      p3.vx = pos.vx + d2.vx * step / 2.0
      p3.vy = pos.vy + d2.vy * step / 2.0
      p3.vz = pos.vz + d2.vz * step / 2.0

      d3 = self.glonass_diffeq(ell_a, gm, j2, omegae_dot, p0, p3)

      p4.x = pos.x + d3.x * step
      p4.y = pos.y + d3.y * step
      p4.z = pos.z + d3.z * step
      p4.vx = pos.vx + d3.vx * step
      p4.vy = pos.vy + d3.vy * step
      p4.vz = pos.vz + d3.vz * step

      d4 = self.glonass_diffeq(ell_a, gm, j2, omegae_dot, p0, p4)

      pos.x += step * (d1.x + 2.0 * d2.x + 2.0 * d3.x + d4.x) / 6.0
      pos.y += step * (d1.y + 2.0 * d2.y + 2.0 * d3.y + d4.y) / 6.0
      pos.z += step * (d1.z + 2.0 * d2.z + 2.0 * d3.z + d4.z) / 6.0
      pos.vx += step * (d1.vx + 2.0 * d2.vx + 2.0 * d3.vx + d4.vx) / 6.0
      pos.vy += step * (d1.vy + 2.0 * d2.vy + 2.0 * d3.vy + d4.vy) / 6.0
      pos.vz += step * (d1.vz + 2.0 * d2.vz + 2.0 * d3.vz + d4.vz) / 6.0

    pos.x -= 0.36
    pos.y += 0.08
    pos.z += 0.18

    travel_time = t - time_tx
    omegatau = omegae_dot * travel_time
    pos.x = pos.x * math.cos(omegatau) + pos.y * math.sin(omegatau)
    pos.y = pos.x * -1 * math.sin(omegatau) + pos.y * math.cos(omegatau)

    return pos.x, pos.y, pos.z, clkcorr

  @timeit #TIME CHECKER!!!
  def glonass_diffeq(self, ell_a, gm, j2, omegae_dot, p0, p):
    p_dot = Rinex.SatPos()
    p_dot.x = p.vx
    p_dot.y = p.vy
    p_dot.z = p.vz

    r2 = p.x * p.x + p.y * p.y + p.z * p.z
    r = math.sqrt(r2)
    g = -gm / (r * r * r)
    h = j2 * 1.5 * ell_a * ell_a / r2
    k = 5.0 * p.z * p.z / r2

    p_dot.vx = g * p.x * (1 - h * (k - 1.0)) + p0.ax + omegae_dot * omegae_dot * p.x + 2.0 * omegae_dot * p.vy
    p_dot.vy = g * p.y * (1 - h * (k - 1.0)) + p0.ay + omegae_dot * omegae_dot * p.y - 2.0 * omegae_dot * p.vx
    p_dot.vz = g * p.z * (1 - h * (k - 3.0)) + p0.az

    return p_dot

  @timeit #TIME CHECKER!!!
  def get_satellite_position_std(self, prn, t, observation):
    omgedo = 7292115.1467e-11
    c = 299792458
    gm = 398.6005e12
    epsec = 1e-12
    maxit = 10

    blk = self.find_nearest_block(prn, t)
    if blk is None:
      return None, None, None, None

    a2 = blk.sqrtA * blk.sqrtA

    travel_time = observation / c
    transmission_time = t - travel_time

    toe_wk = None
    if prn.startswith('C'):
      toe_wk = blk.BDTWeek
    elif prn.startswith('E'):
      toe_wk = blk.GALWeek
    elif prn.startswith('G'):
      toe_wk = blk.GPSWeek
    if prn.startswith('C'):
      toe_wk += 1356
      blk.MJS -= 14
      omgedo -= 1.467e-12
    if not prn.startswith('G'):
      gm -= 5.82e7

    dtc = transmission_time - blk.MJS
    clkcorr = blk.SVclockBias + blk.SVclockDrift * dtc + blk.SVclockDriftRate * dtc * dtc
    if isinstance(blk.name[0], str):
      toe = self.to_mjd_with_week_second(toe_wk, blk.Toe, blk.name[1], prn)
    else:
      toe = self.to_mjd_with_week_second(toe_wk, blk.Toe, blk.name[0], prn)
    dt = transmission_time - toe - clkcorr

    if dt < -Rinex._sec_half_week:
      while dt < -Rinex._sec_half_week:
        dt += Rinex._sec_per_week
    elif dt > Rinex._sec_half_week:
      while dt > Rinex._sec_half_week:
        dt -= Rinex._sec_per_week

    n = math.sqrt(gm/(a2 * a2 * a2)) + blk.DeltaN
    m = blk.M0 + n * dt
    ecc_an = m

    for i in range(maxit):
      old_an = ecc_an
      ecc_an = m + blk.Eccentricity * math.sin(old_an)
      if math.fabs(old_an - ecc_an) < epsec:
        break

    nus = math.sqrt(1 - blk.Eccentricity * blk.Eccentricity) * math.sin(ecc_an)
    nuc = math.cos(ecc_an) - blk.Eccentricity
    nue = 1 - blk.Eccentricity * math.cos(ecc_an)

    nu = math.atan2(nus, nuc)
    phi = nu + blk.omega
    cos2p = math.cos(2 * phi)
    sin2p = math.sin(2 * phi)
    u = phi + blk.Cuc * cos2p + blk.Cus * sin2p
    r = a2 * nue + blk.Crc * cos2p + blk.Crs * sin2p
    cosu = math.cos(u)
    sinu = math.sin(u)

    x_orb = r * cosu
    y_orb = r * sinu

    inc = blk.Io + blk.IDOT * dt + blk.Cic * cos2p + blk.Cis * sin2p
    omeg = blk.Omega0 + blk.OmegaDot * dt
    oedt = omgedo * dt
    if prn in ['C01', 'C02', 'C03', 'C04', 'C05']:
      omeg -= blk.OmegaDot * (blk.Toe + dt)
    else:
      omeg -= omgedo * (blk.Toe + dt)
    cos0 = math.cos(omeg)
    sin0 = math.sin(omeg)
    cosi = math.cos(inc)
    sini = math.sin(inc)
    q1 = -cosi * sin0
    q2 = cosi * cos0

    pos_x = x_orb * cos0 + y_orb * q1
    pos_y = x_orb * sin0 + y_orb * q2
    pos_z = y_orb * sini

    if prn in ['C01', 'C02', 'C03', 'C04', 'C05']:
      temp_pos_x = math.cos(oedt) * pos_x + math.sin(oedt) * 0.9961946980917455 * pos_y + math.sin(
        oedt) * -0.08715574274765817 * pos_z
      temp_pos_y = -math.sin(oedt) * pos_x + math.cos(oedt) * 0.9961946980917455 * pos_y + math.cos(
        oedt) * -0.08715574274765817 * pos_z
      temp_pos_z = 0.08715574274765817 * pos_y + 0.9961946980917455 * pos_z
      pos_x = temp_pos_x
      pos_y = temp_pos_y
      pos_z = temp_pos_z

    cosrr = math.cos(travel_time * omgedo)
    sinrr = math.sin(travel_time * omgedo)
    xs = cosrr * pos_x + sinrr * pos_y
    pos_y = -sinrr * pos_x + cosrr * pos_y
    pos_x = xs

    return pos_x, pos_y, pos_z, clkcorr

  @timeit #TIME CHECKER!!!
  def find_nearest_block(self, prn, t):
    try:
      nav = self.nav[self.nav.index.get_level_values('sv') == prn]
      return nav.iloc[(nav['MJS'] - t).abs().argsort()[0]]
    except:
      return None

  @timeit #TIME CHECKER!!!
  def to_mjd(self, d, prn='G'):
    year_diff = d.year - 1980
    num_leap = math.ceil(year_diff / 4)
    is_leaf = d.year % 4 == 0

    gps_days = year_diff * 365 + num_leap + d.day - 6
    if is_leaf:
      gps_days += Rinex._leap_months[d.month - 1]
    else:
      gps_days += Rinex._normal_months[d.month - 1]
    mjs = Rinex._gpst_0 + gps_days * Rinex._sec_per_day + d.hour * 3600
    mjs += d.minute * 60 + d.second

    mjs = self.adjust_to_gps_time(mjs, d, prn)

    return mjs

  @timeit #TIME CHECKER!!!
  def adjust_to_gps_time(self, mjs, d, prn):
    if prn.startswith('C'):
      mjs += 14
    elif prn.startswith('R'):
      mjs += self.get_leap_second(d)
    return mjs

  @timeit #TIME CHECKER!!!
  def get_leap_second(self, d):
    total_leap = 0
    for l in self.config['etc']['leap_second']:
      if d.date() > l:
        total_leap += 1
    return total_leap

  @timeit #TIME CHECKER!!!
  def to_mjd_with_week_second(self, week, t, d, prn):
    mjs = Rinex._gpst_0 + week * Rinex._sec_per_week + t
    return self.adjust_to_gps_time(mjs, d, prn)

  @classmethod
  @timeit #TIME CHECKER!!!
  def load_config(cls, config='config.yaml'):
    if config is None:
      config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')  
    if not os.path.isfile(config):
      raise FileNotFoundError
    with open(config, 'r') as fp:
      rinex_config = yaml.load(fp.read(), Loader=yaml.FullLoader)
    if rinex_config['rinex']['multiday']['enabled']:
      print('Multiday processing requested.')
      obs_files = os.listdir(rinex_config['rinex']['multiday']['obs_location'])
      nav_files = os.listdir(rinex_config['rinex']['multiday']['nav_location'])
      obs_format = re.compile(rinex_config['rinex']['multiday']['obs_format'].replace('.', '\\.').replace('{DOY}', '(?P<DOY>[0-9]{3})').replace('{YY}', '(?P<YY>[0-9]{2})'))
      csv_location = rinex_config['rinex']['output_dir']
      nav_format = rinex_config['rinex']['multiday']['nav_format']
      for obs in obs_files:
        m = obs_format.search(obs)
        if m is not None:
          print('Processing...', obs)
          obs_search = m.groupdict()
          doy = obs_search['DOY']
          year = int(obs_search['YY']) + 2000
          matching_nav = nav_format.replace('{YYYY}', str(year)).replace('{DOY}', doy).replace('{YY}', str(year - 2000))
          #output_csv = rinex_config['rinex']['multiday']['output'].replace('{DOY}', doy).replace('{YYYY}', str(year)).replace('{YY}', str(year - 2000))
          output_csv = Rinex.output_file_name(csv_location, rinex_config['station']['id'], year, doy)
          if os.path.isfile(os.path.join(rinex_config['rinex']['multiday']['nav_location'], matching_nav)):
            cls([os.path.join(rinex_config['rinex']['multiday']['nav_location'], matching_nav)],
                [os.path.join(rinex_config['rinex']['multiday']['obs_location'], obs)], rinex_config['constellations'],
                year=year, doy=doy).parse()
    else:
      yy, doy, mid = cls.extract_datetime_information(rinex_config['rinex']['singlefile']['observation'][0])
      if yy is None:
        return cls(rinex_config['rinex']['singlefile']['navigation'],
                   rinex_config['rinex']['singlefile']['observation'], rinex_config['constellations'],
                   rinex_config['rinex']['output_dir'])
      year = int(yy) + 2000
      return cls(rinex_config['rinex']['singlefile']['navigation'],
                 rinex_config['rinex']['singlefile']['observation'], rinex_config['constellations'],
                 rinex_config['rinex']['output_dir'], year, doy, mid)

  @classmethod
  @timeit #TIME CHECKER!!!
  def extract_datetime_information(cls, filename):
    filename = os.path.basename(filename) # extract filename from path
    regex_candidates = [re.compile('(?P<station>[A-Za-z0-9]{4})(?P<doy>[0-9]{3})0\.(?P<yy>[0-9]{2})[o|O]'),
                        re.compile('(?P<station>[A-Za-z0-9]{4})(?P<doy>[0-9]{3})(?P<mid>[A-Za-z0-9]{3})\.(?P<yy>[0-9]{2})[o|O]'),
                        re.compile('([A-Za-z0-9]{9})_._(?P<yyyy>[0-9]{4})(?P<doy>[0-9]{3})(?P<hh>[0-9]{2})(?P<mm>[0-9]{2})_([A-Za-z0-9]{3})_([A-Za-z0-9]{3})_([A-Za-z0-9]{2})\.rnx')]
    # return value => [yy, doy, mid|None]
    try:
      # check {SITE}{DOY}0.{YY}O format first
      res = re.compile('(?P<station>[A-Za-z0-9]{4})(?P<doy>[0-9]{3})0\.(?P<yy>[0-9]{2})[o|O]').match(filename).groupdict()
      if len(res) == 3:
        return res['yy'], res['doy'], None
    except:
      ...

    try:
      # check {SITE}{DOY}{MID}.{YY}O format
      res = re.compile('(?P<station>[A-Za-z0-9]{4})(?P<doy>[0-9]{3})(?P<mid>[A-Za-z0-9]{3})\.(?P<yy>[0-9]{2})[o|O]').match(filename).groupdict()
      if len(res) == 4:
        return res['yy'], res['doy'], res['mid']
    except:
      ...

    try:
      # check {SITE}{DOY}{MID}.{YY}O format
      res = re.compile('([A-Za-z0-9]{9})_._(?P<yyyy>[0-9]{4})(?P<doy>[0-9]{3})(?P<hh>[0-9]{2})(?P<mm>[0-9]{2})_([A-Za-z0-9]{3})_([A-Za-z0-9]{3})_([A-Za-z0-9]{2})\.rnx').match(filename).groupdict()
      if len(res) == 4:
        return res['yyyy'][2:], res['doy'], None
    except:
      ...

    return None, None, None


  @classmethod
  @timeit #TIME CHECKER!!!
  def execute_realtime(cls, obs_format, hourly, minute, config=None):
    """ execute rinex realtime
    cls(nav_location, obs_location, constellations, output)
    """
    if config is None:
      config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(config, 'r') as fp:
      rinex_config = yaml.load(fp.read(), Loader=yaml.FullLoader)
    if not os.path.isfile(config):
      raise FileNotFoundError
    if rinex_config['rinex']['realtime']['enabled']:
      start_datetime = datetime.date.fromisoformat(rinex_config['rinex']['multiday']['start_date'])
      end_datetime = datetime.date.fromisoformat(rinex_config['rinex']['multiday']['end_date'])
      doy_str = start_datetime.strftime('%j')
      yyyy_str = str(start_datetime.year)
      hour_minute_idx = hourly + minute
      db_file = cls.output_file_name(rinex_config['rinex']['output_dir'], rinex_config["station"]["id"], yyyy_str, doy_str, hour_minute_index=hour_minute_idx)
      obs_files = os.listdir(rinex_config['rinex']['multiday']['obs_location'])
      nav_files = os.listdir(rinex_config['rinex']['multiday']['nav_location'])
      obs_format = re.compile(obs_format.replace('.', '\\.').replace('{DOY}', '(?P<DOY>[0-9]{3})').replace('{YY}', '(?P<YY>[0-9]{2})'))
      nav_format = rinex_config['rinex']['multiday']['nav_format']
      for obs in obs_files:
        m = obs_format.search(obs)
        if m is not None:
          obs_search = m.groupdict()
          doy = obs_search['DOY']
          year = int(obs_search['YY']) + 2000
          matching_nav = nav_format.replace('{YYYY}', str(year)).replace('{DOY}', doy).replace('{YY}', obs_search['YY'])
          # output_csv = rinex_config['rinex']['multiday']['output'].replace('{DOY}', doy).replace('{YYYY}', str(year)).replace('{YY}', str(year - 2000))
          # output_csv = output_csv[:-4] + '-' + hourly + minute + '.csv'
          if os.path.isfile(os.path.join(rinex_config['rinex']['multiday']['nav_location'], matching_nav)):
            cls([os.path.join(rinex_config['rinex']['multiday']['nav_location'], matching_nav)],
                [os.path.join(rinex_config['rinex']['multiday']['obs_location'], obs)], rinex_config['constellations'],
                db_file, year, doy, hour_minute_idx)
    else:
      yy, doy, mid = cls.extract_datetime_information(rinex_config['rinex']['singlefile']['observation'][0])
      if yy is None:
        return cls(rinex_config['rinex']['singlefile']['navigation'],
                   rinex_config['rinex']['singlefile']['observation'], rinex_config['constellations'],
                   rinex_config['rinex']['output_dir'])
      year = int(yy) + 2000
      return cls(rinex_config['rinex']['singlefile']['navigation'],
                 rinex_config['rinex']['singlefile']['observation'], rinex_config['constellations'],
                 rinex_config['rinex']['output_dir'], year, doy, mid)

  @classmethod
  @timeit #TIME CHECKER!!!
  def output_file_name(cls, base_dir, station_id, yyyy, doy, hour_minute_index=None, postfix=None):
    csv_out_dir = os.path.join(base_dir, station_id, 'compactdb_csv')
    csv_out = f'{yyyy}-{doy}'
    if hour_minute_index is not None:
      csv_out += f'-{hour_minute_index}'
    if postfix is not None:
      csv_out += postfix
    return os.path.join(csv_out_dir, csv_out + '.csv')

  @classmethod
  @timeit #TIME CHECKER!!!
  def load_dir(cls, dir_path):
    if not os.path.isdir(dir_path):
      raise FileNotFoundError(f'{dir_path} not exists')
    nav_file = ''
    obs_file = ''
    for l in os.listdir(dir_path):
      joined_path = os.path.join(dir_path, l)
      if os.path.isfile(joined_path):
        if joined_path.lower().endswith('.rnx'):
          nav_file = joined_path
        elif joined_path.lower().endswith('o'):
          obs_file = joined_path
    return cls(nav_file, obs_file)


if __name__ == '__main__':
  rx = Rinex.load_config()
  if rx is not None:
    rx.parse()
  #rx = Rinex.load_dir('data')
  print_timing_stats() #TIME CHECKER!!!
