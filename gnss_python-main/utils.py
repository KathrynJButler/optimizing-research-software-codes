import math
import pandas as pd
import numpy as np

from scipy.signal import lombscargle


def spread(y, yy, n, x, m, nden):
  """ converted spread function from MATLAB """
  if x == np.round(x):
    x = int(x)
    yy[x - 1] = yy[x - 1] + y
  else:
    ilo = int(np.min([max(np.floor(x - 0.5 * m + 1), 1), n - m + 1]))
    ihi = int(ilo + m - 1)
    fac = (x - ilo) * np.prod([x - i for i in range(ilo + 1, ihi + 1)])
    yy[ihi - 1] = yy[ihi - 1] + y * fac / (nden * (x - ihi))
    for j in range(ihi - 1, ilo - 1, -1):
      nden = (nden / (j + 1- ilo)) * (j - ihi)
      yy[j - 1] = yy[j - 1] + y * fac / (nden * (x - j))
  return yy


def fasper(x, t, ofac, f):
  """ converted fasper function from MATLAB """
  n = len(t)
  tmin = t[0]
  T = t[-1] - tmin
  Ts = T / (n - 1)

  fmax = f
  nout = int(np.round(0.5 * ofac * n))
  f = [x/(n * Ts * ofac) for x in range(1, nout + 1)]
  hifac = fmax / np.max(f)
  MACC = 4
  nfreq = np.power(2, (np.ceil(np.log2(ofac * hifac * n * MACC))))
  fndim = int(2 * nfreq)

  wk1 = [0 for x in range(fndim)]
  wk2 = [0 for x in range(fndim)]

  fac = fndim / (n * Ts * ofac)
  nfac = [1, 1, 2, 6, 24, 720, 5040, 40320, 362880]
  nden = 6

  for j in range(n):
    ck = 1 + np.mod((t[j] - tmin) * fac, fndim)
    ckk = 1 + np.mod(2 * (ck - 1), fndim)
    wk1 = spread(x.iloc[j], wk1, fndim, ck, 4, nden)
    wk2 = spread(1, wk2, fndim, ckk, 4, nden)
  nout = round(0.5 * ofac * hifac * n)
  f = [x/(n*Ts*ofac) for x in range(1, nout + 1)]

  wk1 = np.fft.fft(wk1)
  rwk1 = np.real(wk1[1:nout+1])
  iwk1 = np.imag(wk1[1:nout+1])

  wk2 = np.fft.fft(wk2)
  rwk2 = np.real(wk2[1:nout+1])
  iwk2 = np.imag(wk2[1:nout+1])

  hypo = np.abs(wk2[1:nout+1])
  if 0 in hypo:
    wk2 = lombscargle(x, f, t)
  else:
    hypo = np.abs(wk2[1:nout+1])
    hc2wt = np.divide(0.5 * rwk2, hypo)
    hs2wt = np.divide(0.5 * iwk2, hypo)
    cwt = np.sqrt(0.5 + hc2wt)
    swt = np.multiply(np.sign(hs2wt), np.sqrt(0.5 - hc2wt))
    den = 0.5 * n + np.multiply(hc2wt, rwk2) + np.multiply(hs2wt, iwk2)
    cterm = np.divide(np.power((np.multiply(cwt, rwk1) + np.multiply(swt, iwk1)), 2), den)
    sterm = np.divide(np.power((np.multiply(cwt, iwk1) - np.multiply(swt, rwk1)), 2), n - den)
    wk2 = cterm + sterm

  return wk2, f


def plomb(x, t: pd.core.series.Series, f, ofac):
  """ converted plomb function from MATLAB """
  PNaNPos = [0]

  sig = x
  tout = t

  sig_var = np.var(sig, ddof=1)
  sig_mean = np.mean(sig)

  p = int(np.ceil(np.log2(len(sig))) - 1)
  if len(sig) == (2^p + 1):
    PNaNPos[0] = 2^(p + 1)
  sig = sig - sig_mean
  f = float(f)
  [Px, fr] = fasper(sig, tout, ofac, f)

  P = Px

  bsx_factor = 1/(sig_var * 2)
  P = [x * bsx_factor for x in P]

  return [P, fr]

def mjd2gre(dmjd):
  if np.isnan(dmjd):
    return [np.nan for x in range(6)]
  date = [0 for x in range(6)]

  days = math.floor(dmjd / 86400.0)
  julian = days + 2400001
  time_sec = dmjd - days * 86400.0

  if time_sec < 0.0:
    time_sec = time_sec + 86400.0
    julian -= 1

  jalpha = math.floor(((julian-1867216)-0.25)/36524.25)
  ja = julian + 1 + jalpha - math.floor(0.25 * jalpha)
  jb = ja + 1524
  jc = math.floor((6680.+((jb-2439870)-122.1)/365.25))
  jd = 365 * jc + math.floor(0.25 * jc)
  je = math.floor((jb - jd) / 30.6001)

  date[2] = jb - jd - math.floor(30.6001 * je)
  date[1] = je - 1

  if date[1] > 12:
    date[1] -= 12

  date[0] = jc - 4715

  if date[1] > 2:
    date[0] -= 1

  date[3] = math.floor(time_sec / 3600)
  date[4] = math.floor((time_sec - date[3] * 3600) / 60)
  date[5] = math.floor(time_sec - (date[3] * 3600 + date[4] * 60))

  return date


def gre2mjd(date):
  year = date[0]
  if 80 < year < 100:
    year += 1900
  elif 0 <= year < 80:
    year += 2000

  if date[1] > 2:
    julian_year = year
    julian_month = date[1] + 1
  else:
    julian_year = year - 1
    julian_month = date[1] + 13

  aj = math.floor(julian_year / 100.0)
  julian_date = (math.floor(365.25 * julian_year) + math.floor(30.6001 * julian_month) + date[2] - 679006.0 + 2.0 - aj + math.floor(0.25 * aj)) * 86400.0
  julian_date = julian_date + date[3] * 3600.0 + date[4] * 60 + date[5]

  return julian_date


import os
import mmap
import struct
import platform

class GeoidBadDataFile(Exception):
  pass

class GeoidHeight(object):
  """Calculate the height of the WGS84 geoid above the
  ellipsoid at any given latitude and longitude

  :param name: name to PGM file containing model info
  download from http://geographiclib.sourceforge.net/1.18/geoid.html
  """
  c0 = 240
  c3 = (
    (  9, -18, -88,    0,  96,   90,   0,   0, -60, -20),
    ( -9,  18,   8,    0, -96,   30,   0,   0,  60, -20),
    (  9, -88, -18,   90,  96,    0, -20, -60,   0,   0),
    (186, -42, -42, -150, -96, -150,  60,  60,  60,  60),
    ( 54, 162, -78,   30, -24,  -90, -60,  60, -60,  60),
    ( -9, -32,  18,   30,  24,    0,  20, -60,   0,   0),
    ( -9,   8,  18,   30, -96,    0, -20,  60,   0,   0),
    ( 54, -78, 162,  -90, -24,   30,  60, -60,  60, -60),
    (-54,  78,  78,   90, 144,   90, -60, -60, -60, -60),
    (  9,  -8, -18,  -30, -24,    0,  20,  60,   0,   0),
    ( -9,  18, -32,    0,  24,   30,   0,   0, -60,  20),
    (  9, -18,  -8,    0, -24,  -30,   0,   0,  60,  20),
  )

  c0n = 372
  c3n = (
    (  0, 0, -131, 0,  138,  144, 0,   0, -102, -31),
    (  0, 0,    7, 0, -138,   42, 0,   0,  102, -31),
    ( 62, 0,  -31, 0,    0,  -62, 0,   0,    0,  31),
    (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
    (124, 0,  -62, 0,    0, -124, 0,   0,    0,  62),
    ( 62, 0,  -31, 0,    0,  -62, 0,   0,    0,  31),
    (  0, 0,   45, 0, -183,   -9, 0,  93,   18,   0),
    (  0, 0,  216, 0,   33,   87, 0, -93,   12, -93),
    (  0, 0,  156, 0,  153,   99, 0, -93,  -12, -93),
    (  0, 0,  -45, 0,   -3,    9, 0,  93,  -18,   0),
    (  0, 0,  -55, 0,   48,   42, 0,   0,  -84,  31),
    (  0, 0,   -7, 0,  -48,  -42, 0,   0,   84,  31),
  )

  c0s = 372
  c3s = (
    ( 18,  -36, -122,   0,  120,  135, 0,   0,  -84, -31),
    (-18,   36,   -2,   0, -120,   51, 0,   0,   84, -31),
    ( 36, -165,  -27,  93,  147,   -9, 0, -93,   18,   0),
    (210,   45, -111, -93,  -57, -192, 0,  93,   12,  93),
    (162,  141,  -75, -93, -129, -180, 0,  93,  -12,  93),
    (-36,  -21,   27,  93,   39,    9, 0, -93,  -18,   0),
    (  0,    0,   62,   0,    0,   31, 0,   0,    0, -31),
    (  0,    0,  124,   0,    0,   62, 0,   0,    0, -62),
    (  0,    0,  124,   0,    0,   62, 0,   0,    0, -62),
    (  0,    0,   62,   0,    0,   31, 0,   0,    0, -31),
    (-18,   36,  -64,   0,   66,   51, 0,   0, -102,  31),
    ( 18,  -36,    2,   0,  -66,  -51, 0,   0,  102,  31),
  )

  def __init__(self, name="data/egm2008-1.pgm"):
    self.offset = None
    self.scale = None

    with open(name, "rb") as f:
      line = f.readline()
      if line != b"P5\012" and line != b"P5\015\012":
        raise GeoidBadDataFile("No PGM header")
      headerlen = len(line)
      while True:
        line = f.readline()
        if len(line) == 0:
          raise GeoidBadDataFile("EOF before end of file header")
        headerlen += len(line)
        if line.startswith(b'# Offset '):
          try:
            self.offset = int(line[9:])
          except ValueError as e:
            raise GeoidBadDataFile("Error reading offset", e)
        elif line.startswith(b'# Scale '):
          try:
            self.scale = float(line[8:])
          except ValueError as e:
            raise GeoidBadDataFile("Error reading scale", e)
        elif not line.startswith(b'#'):
          try:
            self.width, self.height = list(map(int, line.split()))
          except ValueError as e:
            raise GeoidBadDataFile("Bad PGM width&height line", e)
          break
      line = f.readline()
      headerlen += len(line)
      levels = int(line)
      if levels != 65535:
        raise GeoidBadDataFile("PGM file must have 65535 gray levels")
      if self.offset is None:
        raise GeoidBadDataFile("PGM file does not contain offset")
      if self.scale is None:
        raise GeoidBadDataFile("PGM file does not contain scale")

      if self.width < 2 or self.height < 2:
        raise GeoidBadDataFile("Raster size too small")

      fd = f.fileno()
      fullsize = os.fstat(fd).st_size

      if fullsize - headerlen != self.width * self.height * 2:
        raise GeoidBadDataFile("File has the wrong length")

      self.headerlen = headerlen
      if platform.system() == 'Windows':
        self.raw = mmap.mmap(fd, fullsize, access=mmap.ACCESS_READ)
      else:
        self.raw = mmap.mmap(fd, fullsize, mmap.MAP_SHARED, mmap.PROT_READ)

    self.rlonres = self.width / 360.0
    self.rlatres = (self.height - 1) / 180.0
    self.ix = None
    self.iy = None

  def _rawval(self, ix, iy):
    if iy < 0:
      iy = -iy
      ix += self.width/2
    elif iy >= self.height:
      iy = 2 * (self.height - 1) - iy
      ix += self.width/2
    if ix < 0:
      ix += self.width
    elif ix >= self.width:
      ix -= self.width

    return struct.unpack_from('>H', self.raw,
                              (iy * self.width + ix) * 2 + self.headerlen
                              )[0]

  def get(self, lat, lon, cubic=True):
    if lon < 0:
      lon += 360
    fy = (90 - lat) * self.rlatres
    fx = lon * self.rlonres
    iy = int(fy)
    ix = int(fx)
    fx -= ix
    fy -= iy
    if iy == self.height - 1:
      iy -= 1

    if ix != self.ix or iy != self.iy:
      self.ix = ix
      self.iy = iy
      if not cubic:
        self.v00 = self._rawval(ix, iy)
        self.v01 = self._rawval(ix+1, iy)
        self.v10 = self._rawval(ix, iy+1)
        self.v11 = self._rawval(ix+1, iy+1)
      else:
        v = (
          self._rawval(ix    , iy - 1),
          self._rawval(ix + 1, iy - 1),
          self._rawval(ix - 1, iy    ),
          self._rawval(ix    , iy    ),
          self._rawval(ix + 1, iy    ),
          self._rawval(ix + 2, iy    ),
          self._rawval(ix - 1, iy + 1),
          self._rawval(ix    , iy + 1),
          self._rawval(ix + 1, iy + 1),
          self._rawval(ix + 2, iy + 1),
          self._rawval(ix    , iy + 2),
          self._rawval(ix + 1, iy + 2)
        )
        if iy == 0:
          c3x = GeoidHeight.c3n
          c0x = GeoidHeight.c0n
        elif iy == self.height - 2:
          c3x = GeoidHeight.c3s
          c0x = GeoidHeight.c0s
        else:
          c3x = GeoidHeight.c3
          c0x = GeoidHeight.c0
        self.t = [
          sum([ v[j] * c3x[j][i] for j in range(12) ]) / float(c0x)
          for i in range(10)
        ]
    if not cubic:
      a = (1 - fx) * self.v00 + fx * self.v01
      b = (1 - fx) * self.v10 + fx * self.v11
      h = (1 - fy) * a + fy * b
    else:
      h = (
          self.t[0] +
          fx * (self.t[1] + fx * (self.t[3] + fx * self.t[6])) +
          fy * (
              self.t[2] + fx * (self.t[4] + fx * self.t[7]) +
              fy * (self.t[5] + fx * self.t[8] + fy * self.t[9])
          )
      )
    return self.offset + self.scale * h


if __name__ == '__main__':
  pass