from veros.core.operators import numpy as npx


def c_point_to_z_point(state,Cfield, noSlip = True):

    '''calculates value at z-point by averaging c-point values'''

    sumNorm = state.variables.iceMask + npx.roll(state.variables.iceMask,1,1)
    sumNorm = sumNorm + npx.roll(sumNorm,1,0)
    if noSlip:
        sumNorm = npx.where(sumNorm>0,1./sumNorm,0.)
    else:
        sumNorm = npx.where(sumNorm==4.,0.25,0.)

    Zfield =             Cfield + npx.roll(Cfield,1,1)
    Zfield = sumNorm * ( Zfield + npx.roll(Zfield,1,0) )

    return Zfield