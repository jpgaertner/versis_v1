from veros import veros_routine
from versis.parameters import heatCapacity, rhoFresh

def copy_output(state):
    vs = state.variables

    vs.forc_salt_surface += vs.saltflux + vs.EmPmR * vs.ocSalt

    # set the surface heat flux to the heat flux that is reduced due to a potential ice cover
    vs.forc_temp_surface = - vs.Qnet / ( heatCapacity * rhoFresh )

