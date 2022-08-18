from veros.state import VerosState
from veros.settings import Setting
from veros.variables import Variable
from veros import veros_routine
from veros.core.operators import numpy as npx
from veros.core.operators import update, at

nx = 65
ny = nx
nITC = 1
olx = 2
oly = 2

from parameters import *

from fill_overlap import fill_overlap, fill_overlap3d

from settings import SETTINGS

dimensions = dict(x=nx+2*olx, y=ny+2*oly, z=nITC)
dim = ("x","y")

VARIABLES = dict(
    hIceMean        = Variable("Mean ice thickness", dim, "m"),
    hSnowMean       = Variable("Mean snow thickness", dim, "m"),
    Area            = Variable("Sea ice cover fraction", dim, " "),
    TIceSnow        = Variable("Ice/ snow temperature", ("x","y","z"), "K"),
    uIce            = Variable("Zonal ice velocity", dim, "m/s"),
    vIce            = Variable("Merdidional ice velocity", dim, "m/s"),
    uOcean            = Variable("Zonal ocean surface velocity", dim, "m/s"),
    vOcean            = Variable("Merdional ocean surface velocity", dim, "m/s"),
    uWind           = Variable("Zonal wind velocity", dim, "m/s"),
    vWind           = Variable("Merdional wind velocity", dim, "m/s"),
    wSpeed          = Variable("Total wind speed", dim, "m/s"),
    fu              = Variable("Zonal stress on ocean surface", dim, "N/m^2"),
    fv              = Variable("Meridional stress on ocean surface", dim, "N/m^2"),
    WindForcingX    = Variable("Zonal forcing on ice by wind stress", dim, "N"),
    WindForcingY    = Variable("Meridional forcing on ice by wind stress", dim, "N"),
    etaN            = Variable("Ocean surface height anomaly", dim, "m"),
    surfPress       = Variable("Surface pressure", dim, "P"),
    SeaIceLoad      = Variable("Load of sea ice on ocean surface", dim, "kg/m^2"),
    ocSalt        = Variable("Ocean surface salinity", dim, "g/kg"),
    theta           = Variable("Ocean surface temperature", dim, "K"),
    Qnet            = Variable("Net heat flux out of the ocean", dim, "W/m^2"),
    Qsw             = Variable("Shortwave heatflux into the ocean", dim, "W/m^2"),
    SWDown          = Variable("Downward shortwave radiation", dim, "W/m^2"),
    LWDown          = Variable("Downward longwave radiation", dim, "W/m^2"),
    ATemp           = Variable("Atmospheric temperature", dim, "K"),
    aqh             = Variable("Atmospheric specific humidity", dim, "g/kg"),
    precip          = Variable("Precipitation rate", dim, "m/s"),
    snowfall        = Variable("Snowfall rate", dim, "m/s"),
    #TODO: how to convert evaporation to evaporation rate?
    evap            = Variable("Evaporation rate over open ocean", dim, "m/s"),
    runoff          = Variable("Runoff into ocean", dim, "m/s"),
    EmPmR           = Variable("Evaporation minus precipitation minus runoff", dim, "m/s"),
    saltflux        = Variable("Salt flux into the ocean", dim, "m/s"),
    R_low           = Variable("Sea floor depth (<0)", dim, "m"),
    ssh_an          = Variable("Sea surface height anomaly", dim, "m"),
    SeaIceMassC     = Variable("Sea ice mass centered around c point", dim, "kg"),
    SeaIceMassU     = Variable("Sea ice mass centered around u point", dim, "kg"),
    SeaIceMassV     = Variable("Sea ice mass centered around v point", dim, "kg"),
    SeaIceStrength  = Variable("Ice Strength", dim, "N/m"),
    zeta            = Variable("Bulk ice viscosity", dim, "Ns/m^2"),
    eta             = Variable("Shear ice viscosity", dim, "Ns/m^2"),
    os_hIceMean     = Variable("Overshoot of ice thickness from advection", dim, "m"),
    os_hSnowMean    = Variable("Overshoot of snow thickness from advection", dim, "m/s"),
    AreaW           = Variable("Sea ice cover fraction centered around u point", dim, " "),
    AreaS           = Variable("Sea ice cover fraction centered around v point", dim, " "),
    sigma11         = Variable("Stress tensor element", dim, "N/m^2"),
    sigma22         = Variable("Stress tensor element", dim, "N/m^2"),
    sigma12         = Variable("Stress tensor element", dim, "N/m^2"),
    tauX            = Variable("Zonal surface stress", dim, "N/m^2"),
    tauY            = Variable("Meridional surface stress", dim, "N/m^2"),
    recip_hIceMean  = Variable("1 / hIceMean", dim, "1/m"),
    fCori           = Variable("Coriolis parameter", dim, "1/s"),
    k1AtC           = Variable(" ", dim, " "),
    k2AtC           = Variable(" ", dim, " "),
    k1AtZ           = Variable(" ", dim, " "),
    k2AtZ           = Variable(" ", dim, " "),
    maskInC         = Variable("Mask at c-points", dim, " "),
    maskInU         = Variable("Mask at u-points", dim, " "),
    maskInV         = Variable("Mask at v-points", dim, " "),
    iceMask         = Variable("??", dim, " "), #TODO what was this again?
    iceMaskU     = Variable("??", dim, " "),
    iceMaskV     = Variable("??", dim, " "),
    dxC             = Variable("zonal spacing of cell centers across western cell wall", dim, "[m]"),
    dyC             = Variable("meridional spacing of cell centers across southern cell wall", dim, "[m]"),
    dxG             = Variable("zonal spacing of cell faces along southern cell wall", dim, "[m]"),
    dyG             = Variable("meridional spacing of cell faces along western cell wall", dim, "[m]"),
    dxU             = Variable("zonal spacing of u-points through cell center", dim, "[m]"),
    dyU             = Variable("meridional spacing of u-points through south-west corner of the cel", dim, "[m]"),
    dxV             = Variable("zonal spacing of v-points through south-west corner of the cell", dim, "[m]"),
    dyV             = Variable("meridional spacing of v-points through cell center", dim, "[m]"),
    recip_dxC       = Variable("1 / dxC", dim, "[1/m]"),
    recip_dyC       = Variable("1 / dyC", dim, "[1/m]"),
    recip_dxG       = Variable("1 / dxG", dim, "[1/m]"),
    recip_dyG       = Variable("1 / dyG", dim, "[1/m]"),
    recip_dxU       = Variable("1 / dxU", dim, "[1/m]"),
    recip_dyU       = Variable("1 / dyU", dim, "[1/m]"),
    recip_dxV       = Variable("1 / dxV", dim, "[1/m]"),
    recip_dyV       = Variable("1 / dyV", dim, "[1/m]"),
    rA              = Variable("grid cell area centered around c-point", dim, "[m^2]"),
    rAu             = Variable("grid cell area centered around u-point", dim, "[m^2]"),
    rAv             = Variable("grid cell area centered around v-point", dim, "[m^2]"),
    rAz             = Variable("grid cell area centered around z-point", dim, "[m^2]"),
    recip_rA        = Variable("1 / rA", dim, ["1/m^2"]),
    recip_rAu       = Variable("1 / rAu", dim, ["1/m^2"]),
    recip_rAv       = Variable("1 / rAv", dim, ["1/m^2"]),
    recip_rAz       = Variable("1 / rAz", dim, ["1/m^2"]),
)

ones2d = npx.ones((nx+2*olx,ny+2*oly))
ones3d = npx.ones((nx+2*olx,ny+2*oly,nITC))
onesWind = npx.ones((32,nx+2*olx,ny+2*oly))
def copy(x):
    return update(x, at[:,:], x)

@veros_routine
def set_init_values(state):
    state.variables.hIceMean    = ones2d * 1.3   #hIce_gen
    state.variables.hSnowMean   = ones2d * 0.1   #ones2d * 0
    state.variables.Area        = ones2d * 0.9   #ones2d * 1
    state.variables.TIceSnow    = ones3d * 273.0
    state.variables.SeaIceLoad  = ones2d * (rhoIce * state.variables.hIceMean
                                            + rhoSnow * state.variables.hSnowMean)
    state.variables.uWind       = ones2d * 1
    state.variables.vWind       = ones2d * 1
    state.variables.wSpeed      = ones2d * 2     #npx.sqrt(state.variables.uWind**2 + state.variables.vWind**2)
    state.variables.uOcean        = ones2d * 1
    state.variables.vOcean        = ones2d * 1
    state.variables.ocSalt    = ones2d * 29
    state.variables.theta       = ones2d * celsius2K - 1.66
    state.variables.Qnet        = ones2d * 173.03212617345582
    state.variables.Qsw         = ones2d * 0
    state.variables.SWDown      = ones2d * 0
    state.variables.LWDown      = ones2d * 80
    state.variables.ATemp       = ones2d * 253
    state.variables.R_low       = ones2d * -1000
    state.variables.R_low = update(state.variables.R_low, at[:,-1], 0)
    state.variables.R_low = update(state.variables.R_low, at[-1,:], 0)
    state.variables.precip      = ones2d * 0


from set_inits import set_inits

state = VerosState(VARIABLES, SETTINGS, dimensions)
state.initialize_variables()
set_inits(state)
set_init_values(state)

# reciprocal of timesteps
recip_deltatTherm = 1 / state.settings.deltatTherm
recip_deltatDyn   = 1 / state.settings.deltatDyn