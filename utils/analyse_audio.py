import numpy as np
from pycochleagram import cochleagram as cgram

def get_cochleagram(signal, sr, n, sample_factor=2, downsample=None, nonlinearity=None):
  """Demo the cochleagram generation.

    signal (array): If a time-domain signal is provided, its
      cochleagram will be generated with some sensible parameters. If this is
      None, a synthesized tone (harmonic stack of the first 40 harmonics) will
      be used.
    sr: (int): If `signal` is not None, this is the sampling rate
      associated with the signal.
    n (int): number of filters to use.
    sample_factor (int): Determines the density (or "overcompleteness") of the
      filterbank. Original MATLAB code supported 1, 2, 4.
    downsample({None, int, callable}, optional): Determines downsampling method to apply.
      If None, no downsampling will be applied. If this is an int, it will be
      interpreted as the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor). A custom downsampling function can
      be provided as a callable. The callable will be called on the subband
      envelopes.
    nonlinearity({None, 'db', 'power', callable}, optional): Determines
      nonlinearity method to apply. None applies no nonlinearity. 'db' will
      convert output to decibels (truncated at -60). 'power' will apply 3/10
      power compression.

    Returns:
      array:
        **cochleagram**: The cochleagram of the input signal, created with
          largely default parameters.
  """
  human_coch = cgram.human_cochleagram(signal, sr, n=n, sample_factor=sample_factor,
      downsample=downsample, nonlinearity=nonlinearity, strict=False)
  img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)
  return img

# def get_log_gamma(signal):
#     import brian2hears as b2h
#     from brian2 import Hz, kHz

#     nbr_center_frequencies = 50  #number of frequency channels in the filterbank

#     c1 = -2.96 #glide slope
#     b1 = 1.81  #factor determining the time constant of the filters

#     #center frequencies with a spacing following an ERB scale
#     cf = b2h.erbspace(100*Hz, 1000*Hz, nbr_center_frequencies)

#     gamma_chirp = b2h.LogGammachirp(signal, cf, c=c1, b=b1)

#     gamma_chirp_mon = gamma_chirp.process()

#     return np.flipud(gamma_chirp_mon)