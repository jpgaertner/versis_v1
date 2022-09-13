from veros import veros_routine
from veros.core.operators import numpy as npx
from versis.parameters import heatCapacity, rhoSea, celsius2K, gravity

@veros_routine
def copy_input(state):
    vs = state.variables

    ###### ocean forcing #####

    vs.uOcean = vs.u[:,:,-1,vs.tau]
    vs.vOcean = vs.v[:,:,-1,vs.tau]

    vs.theta = vs.temp[:,:,-1,vs.tau] + celsius2K
    vs.ocSalt = vs.salt[:,:,-1,vs.tau]

    vs.R_low = vs.ht

    vs.Qnet = - vs.forc_temp_surface * heatCapacity * rhoSea

    vs.Qsw = - vs.SWDown


@veros_routine
def copy_output(state):
    vs = state.variables

    # salt flux into the ocean
    #vs.forc_salt_surface = vs.saltflux + vs.EmPmR

    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( heatCapacity * rhoSea )

    # update the ocean surface stress so the total stress is used
    vs.surface_taux = state.variables.surface_taux * (1 - state.variables.AreaW) \
                    + state.variables.fu * state.variables.AreaW
    vs.surface_tauy = state.variables.surface_tauy * (1 - state.variables.AreaS) \
                    + state.variables.fv * state.variables.AreaS