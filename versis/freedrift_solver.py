from veros.core.operators import numpy as npx
from veros import veros_kernel

from versis.parameters import rhoIce, rhoSea, waterIceDrag, waterIceDrag_south


@veros_kernel
def freedrift_solver(state):

    '''calculate ice velocities without taking into account internal ice stress'''

    # air-ice stress at c-point
    tauXIceCenter = 0.5 * (state.variables.WindForcingX \
                + npx.roll(state.variables.WindForcingX,-1,1))
    tauYIceCenter = 0.5 * (state.variables.WindForcingY \
                + npx.roll(state.variables.WindForcingY,-1,0))

    # mass of ice per unit area times coriolis factor
    mIceCor = rhoIce * state.variables.hIceMean * state.variables.fCori

    # ocean surface velocity at c-points
    uOceanCenter = 0.5 * (state.variables.uOcean + npx.roll(state.variables.uOcean,-1,1))
    vOceanCenter = 0.5 * (state.variables.vOcean + npx.roll(state.variables.vOcean,-1,0))

    # right hand side of the free drift equation
    rhsX = - tauXIceCenter - mIceCor * vOceanCenter
    rhsY = - tauYIceCenter + mIceCor * uOceanCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    where1 = (tmp1 > 0)
    rhsN = npx.where(where1, npx.sqrt(rhsX**2 + rhsY**2), 0)
    rhsA = npx.where(where1, npx.arctan2(rhsY,rhsX), 0)

    # solve for norm
    south = (state.variables.fCori < 0)
    tmp1 = 1 / (npx.where(south, waterIceDrag_south, waterIceDrag) * rhoSea)
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = npx.where(tmp3 > 0, npx.sqrt(0.5 * (npx.sqrt(tmp4) - tmp2)), 0)

    # solve for angle
    tmp1 = 1 / tmp1
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = npx.where(tmp4 > 0, rhsA - npx.arctan2(tmp3, tmp2), 0)

    # calculate velocities at c-points
    uIceCenter = uOceanCenter - solNorm * npx.cos(solAngle)
    vIceCenter = vOceanCenter - solNorm * npx.sin(solAngle)

    # interpolate to velocity points
    uIceFD = 0.5 * (npx.roll(uIceCenter,1,1) + uIceCenter)
    vIceFD = 0.5 * (npx.roll(vIceCenter,1,0) + vIceCenter)

    # apply masks
    uIceFD = uIceFD * state.variables.iceMaskU
    vIceFD = vIceFD * state.variables.iceMaskV

    return uIceFD, vIceFD