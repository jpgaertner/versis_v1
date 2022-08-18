from veros import veros_routine
from veros.core.operators import numpy as npx
from veros.core.operators import update, at

from versis.fill_overlap import fill_overlap
from versis.parameters import celsius2K


@veros_routine
def set_inits(state):
    vs = state.variables
    st = state.settings

    vs.dxC = npx.ones_like(vs.iceMask) * st.gridcellWidth
    vs.dyC = vs.dxC
    vs.dxG = vs.dxC
    vs.dyG = vs.dxC
    vs.dxU = vs.dxC
    vs.dyU = vs.dxC
    vs.dxV = vs.dxC
    vs.dyV = vs.dxC

    vs.recip_dxC = 1 / vs.dxC
    vs.recip_dyC = 1 / vs.dyC
    vs.recip_dxG = 1 / vs.dxG
    vs.recip_dyG = 1 / vs.dyG
    vs.recip_dxU = 1 / vs.dxU
    vs.recip_dyU = 1 / vs.dyU
    vs.recip_dxV = 1 / vs.dxV
    vs.recip_dyV = 1 / vs.dyV

    vs.rA = vs.dxU * vs.dyV
    vs.rAz = vs.dxV * vs.dyU
    vs.rAu = vs.dxC * vs.dyG
    vs.rAv = vs.dxG * vs.dyC

    vs.maskInC = npx.ones_like(vs.iceMask)
    vs.maskInC = update(vs.maskInC, at[-st.oly-1,:], 0)
    vs.maskInC = update(vs.maskInC, at[:,-st.olx-1], 0)
    vs.maskInU = vs.maskInC * npx.roll(vs.maskInC,1,axis=1)
    vs.maskInU = fill_overlap(state,vs.maskInU)
    vs.maskInV = vs.maskInC * npx.roll(vs.maskInC,1,axis=0)
    vs.maskInV = fill_overlap(state,vs.maskInV)

    vs.iceMask = vs.maskInC
    vs.iceMaskU = vs.maskInU
    vs.iceMaskV = vs.maskInV


    vs.hIceMean = npx.ones_like(vs.iceMask) * 1.3
    vs.hSnowMean = npx.ones_like(vs.iceMask) * 0.1
    vs.Area = npx.ones_like(vs.iceMask) * 0.9
    vs.TIceSnow = npx.ones((*vs.iceMask.shape,st.nITC)) * 273
    # vs.TIceSnow = npx.ones((*vs.iceMask.shape,15)) * 273
    vs.wSpeed = npx.ones_like(vs.iceMask) * 2#5
    vs.ocSalt = npx.ones_like(vs.iceMask) * 29
    vs.theta = npx.ones_like(vs.iceMask) * celsius2K - 2
    vs.Qnet = npx.ones_like(vs.iceMask) * 168.99925638039838
    vs.LWDown = npx.ones_like(vs.iceMask) * 80
    vs.ATemp = npx.ones_like(vs.iceMask) * 243