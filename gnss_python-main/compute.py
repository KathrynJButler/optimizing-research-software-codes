from scipy.signal import find_peaks
from utils import *
import math

def ECEFtoGPS(pos):
  x, y, z = pos[0], pos[1], pos[2]

  a = 6378137
  e = 8.1819190842622e-2

  b = math.sqrt(math.pow(a, 2) * (1 - math.pow(e, 2)))
  ep = math.sqrt((math.pow(a, 2) - math.pow(b, 2)) / pow(b, 2))
  p = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
  th = math.atan2(a * z, b * p)
  lon = math.atan2(y, x)
  lat = math.atan((z + math.pow(ep, 2) * b * math.pow(math.sin(th), 3)) / (p - math.pow(e, 2) * a * math.pow(math.cos(th), 3)))

  N = a / math.sqrt(1 - math.pow(e, 2) * math.pow(math.sin(lat), 2))
  alt = p / math.cos(lat) - N

  if math.fabs(x) < 1 and math.fabs(y) < 1:
    alt = math.fabs(z) - b
  return [lat, lon, alt]


def XYZtoENU(a, p, l):
  enu = np.array([
    [-math.sin(l), math.cos(l), 0],
    [-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)],
    [math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p)]
  ])
  return np.dot(enu, a.reshape((3, 1)))


def calculate_elevation_azimuth(recv, sat) -> list:
  r = sat - recv
  gps = ECEFtoGPS(recv)
  enu = XYZtoENU(r, gps[0], gps[1])
  enu_norm = np.linalg.norm(enu)
  elevation = math.asin(enu[2] / enu_norm)
  azimuth = math.atan2(enu[0] / enu_norm, enu[1] / enu_norm)

  elevation = math.degrees(elevation)
  azimuth = math.degrees(azimuth)
  if azimuth < 0:
    azimuth += 360

  return [elevation, azimuth]


def antenna_height_gps(prn, rx, config):
  mean_time = rx.data_frame['MJS Timestamp'].mean()
  gre = mjd2gre(mean_time)
  time_mjd1 = int(gre2mjd([gre[0], gre[1], gre[2], 0, 0, 0]))
  time_mjd2 = int(gre2mjd([gre[0], gre[1], gre[2], 24, 0, 0]))

  data_frame = rx.data_frame.query(f'SatID == "{prn}"')

  xtick = [x for x in range(time_mjd1, time_mjd2, int(86400/8))]

  for idx, row in data_frame.iterrows():
    elevation, azimuth = calculate_elevation_azimuth(np.array(config['station']['info']),
                                                     np.array([row['PosX'], row['PosY'], row['PosZ']]))
    # elevation = math.degrees(elevation)
    # azimuth = math.degrees(azimuth)
    # if azimuth < 0:
    #   azimuth += 360
    data_frame['AziAngle'][idx] = azimuth
    data_frame['ElevAngle'][idx] = elevation

  data_frame = data_frame.query('ElevAngle >= 0')  # keep only ElevAngle is greater than 0
  data_frame = data_frame[['MJS Timestamp', 'PosX', 'PosY', 'PosZ', 'AziAngle', 'ElevAngle']]
  data_frame = data_frame.drop_duplicates()

  for mask in config['mask']['azimuth']:
    data_frame = data_frame.query(f'AziAngle < {mask[0]} | AziAngle >= {mask[1]}')


  for mask in config['mask']['elevation']:
    data_frame = data_frame.query(f'ElevAngle < {mask[0]} | ElevAngle >= {mask[1]}')

  if len(data_frame) < 3:
    return None

  local_maxima, _ = find_peaks(data_frame['ElevAngle'])
  local_minima, _ = find_peaks(-1 * data_frame['ElevAngle']) * 1

  peaks = []

  if len(local_maxima) > 0:
    for m in local_maxima:
      pass
  if len(local_minima) > 0:
    # if local minima exists
    pass

  discont_idx = []
  for i in range(len(data_frame['MJS Timestamp']) - 1):
    if data_frame['MJS Timestamp'].iloc[i + 1] - data_frame['MJS Timestamp'].iloc[i] > 600:
      discont_idx.append(i)
      discont_idx.append(i + 1)
  peaks += discont_idx
  peaks = list(set(peaks))
  peaks = sorted(peaks)

  peaks = list(set([0] + peaks + [len(data_frame) - 1])) # last index-1 because python list index starts from 0

  for i in range(len(peaks)):
    arc_time_length = (data_frame['MJS Timestamp'].iloc[peaks[i + 1]] - data_frame['MJS Timestamp'].iloc[peaks[i]]) / 60
    if arc_time_length < config['mask']['slid']['max']:
      continue
    peak_time = data_frame['MJS Timestamp'].iloc[peaks[0]]
    div_str = [peak_time,
               peak_time + config['mask']['slid']['min']*60,
               peak_time + (arc_time_length - config['mask']['slid']['max']) * 60]
    div_end = [x + config['mask']['slid']['max'] * 60 for x in div_str]

    div_idx_str = []
    div_idx_end = []
    div_idx_str_end = []
    for j in range(len(div_str)):
      div_idx_str.append(data_frame.query(f'`MJS Timestamp` <= {div_str[j]}')['MJS Timestamp'].max())
      div_idx_end.append(data_frame.query(f'`MJS Timestamp` <= {div_end[j]}')['MJS Timestamp'].max())
      """
      try using idxmax() instead of max()
      idx = data_frame.query(f'`MJS Timestamp` <= {div_end[j]}')['MJS Timestamp'].idxmax()
      # can be accessed with index
      data_frame[data_frame.index == idx]
      """
      div_idx_str_end.append([div_idx_str[-1], div_idx_end[-1]])



