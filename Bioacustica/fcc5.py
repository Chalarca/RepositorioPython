import numpy as np
import numpy.matlib
import scipy.io as sio
import skfuzzy as fuzz


def fcc5(canto,nfiltros,nc,nframes):    
    """ Calculo de FCCs con escalamiento lineal """
   # nfiltros = 14
    #nc = 4
    #nframes = 4
    #I11 = sio.loadmat('MCanto.mat')
    #canto = I11['canto']
    a, b = np.shape(canto)
    div = nframes
    w = int(np.floor(b/div))
    b1 = np.empty((a, 0), float)
    for k in range(0, w*div, w):
        bb = np.transpose(np.expand_dims(np.sum(np.power
             (np.abs(canto[:, k:k + w]), 2), axis=1), axis=0))
        b1 = np.append(b1, bb, axis=1)

    if a >= nfiltros:
        _h = np.zeros((nfiltros, a), np.double)
        wf = int(np.floor(a/nfiltros))
        h = np.empty((0, a), float)
        for k in range(0, wf*nfiltros, wf):
            hh = np.expand_dims(fuzz.gaussmf
                 (np.arange(a) + 1, k + wf, wf/4), axis=0)
            h= np.append(h, hh, axis=0)
    fbe = h@b1
    n = nc
    m = nfiltros

    dctm = lambda n, m: np.multiply(np.sqrt(2/m),
           np.cos(np.multiply(np.matlib.repmat
           (np.transpose(np.expand_dims
           (np.arange(n), axis=0)), 1, m),
           np.matlib.repmat(np.expand_dims
           (np.multiply(np.pi, np.arange
           (1, m + 1)-0.5)/m, axis=0), n, 1))))
    dct = dctm(n, m)
    y = dct@np.log(fbe)
    return y