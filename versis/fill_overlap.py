from veros.core.operators import update, at
from veros import veros_kernel


@veros_kernel
def fill_overlap(state, A):
    '''
    fills the overlaps of a field of size (ny+2*oly, nx+2*olx) where
    [oly:-oly,olx:-olx] is the actual cell (requires n > ol)
    '''

    olx = state.settings.olx
    oly = state.settings.oly
    A = update(A, at[:oly,:], A[-2*oly:-oly,:])
    A = update(A, at[-oly:,:], A[oly:2*oly,:])
    A = update(A, at[:,:olx], A[:,-2*olx:-olx])
    A = update(A, at[:,-olx:], A[:,olx:2*olx])

    return A

@veros_kernel
def fill_overlap3d(state, A):
    olx = state.settings.olx
    oly = state.settings.oly
    A = update(A, at[:,:oly,:], A[:,-2*oly:-oly,:])
    A = update(A, at[:,-oly:,:], A[:,oly:2*oly,:])
    A = update(A, at[:,:,:olx], A[:,:,-2*olx:-olx])
    A = update(A, at[:,:,-olx:], A[:,:,olx:2*olx])

    return A

@veros_kernel
def fill_overlap_uv(state, U, V):
    return fill_overlap(state, U), fill_overlap(state, V)
