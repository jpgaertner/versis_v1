from veros import veros_routine
from veros.core.operators import numpy as npx
from versis.parameters import heatCapacity, rhoFresh, celsius2K

@veros_routine
def copy_input(state):
    vs = state.variables

    ###### ocean forcing #####

    vs.uOcean = vs.u[:,:,-1,vs.tau]
    vs.vOcean = vs.v[:,:,-1,vs.tau]

    vs.theta = vs.temp[:,:,-1,vs.tau] + celsius2K
    vs.ocSalt = vs.salt[:,:,-1,vs.tau]

    vs.R_low = vs.ht


    ### atmospheric forcing ###

    vs.Qnet = - vs.forc_temp_surface * heatCapacity * rhoFresh
    vs.Qsw = - vs.SWDown


    ##### masks #####

    vs.iceMask = vs.maskT[:,:,-1]
    vs.iceMaskU = vs.maskU[:,:,-1]
    vs.iceMaskV = vs.maskV[:,:,-1]
    vs.maskInC = vs.iceMask
    vs.maskInU = vs.iceMaskU
    vs.maskInV = vs.iceMaskV


    ##### grid #####

    vs.fCori = vs.coriolis_t
    vs.dxC = npx.ones_like(vs.dxC) * vs.dxt[:,npx.newaxis]
    vs.dyC = npx.ones_like(vs.dxC) * vs.dyt
    vs.dxU = npx.ones_like(vs.dxC) * vs.dxu[:,npx.newaxis]
    vs.dyU = npx.ones_like(vs.dxC) * vs.dyu

    # these are not specified in veros
    vs.dxG = npx.ones_like(vs.dxC) * vs.dxU
    vs.dyG = npx.ones_like(vs.dxC) * vs.dyU
    vs.dxV = npx.ones_like(vs.dxC) * vs.dxU
    vs.dyV = npx.ones_like(vs.dxC) * vs.dyU

    vs.recip_dxC = 1 / vs.dxC
    vs.recip_dyC = 1 / vs.dyC
    vs.recip_dxG = 1 / vs.dxG
    vs.recip_dyG = 1 / vs.dyG
    vs.recip_dxU = 1 / vs.dxU
    vs.recip_dyU = 1 / vs.dyU
    vs.recip_dxV = 1 / vs.dxV
    vs.recip_dyV = 1 / vs.dyV

    vs.rA = vs.area_t
    vs.rAu = vs.area_u
    vs.rAv = vs.area_v
    vs.rAz = vs.rA
    vs.recip_rA = 1 / vs.rA
    vs.recip_rAu = 1 / vs.rAu
    vs.recip_rAv = 1 / vs.rAv
    vs.recip_rAz = 1 / vs.rAz

@veros_routine
def copy_output(state):
    vs = state.variables

    vs.forc_salt_surface += vs.saltflux + vs.EmPmR * vs.ocSalt

    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( heatCapacity * rhoFresh )