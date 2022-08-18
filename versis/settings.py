from veros.settings import Setting


SETTINGS = dict(
    use_seaice              = Setting(False, bool, "flag for using the sea ice plug in"),
    deltatTherm             = Setting(86400, float, "Timestep for thermodynamic equations [s]"),
    recip_deltatTherm       = Setting(1/86400, float, "1 / deltatTherm [1/s]"),
    deltatDyn               = Setting(86400, float, "Timestep for dynamic equations [s]"),
    nx                      = Setting(65, int, "Grid points in zonal direction"),
    ny                      = Setting(65, int, "Grid points in meridional direction"),
    olx                     = Setting(2, int, "Grid points in zonal overlap"),
    oly                     = Setting(2, int, "Grid points in meridional overlap"),
    gridcellWidth           = Setting(8000, float, "Grid cell width [m]"),
    nITC                    = Setting(1, int, "Number of ice thickness categories"),
    recip_nITC              = Setting(1, int, "1 / nITC"),
    noSlip                  = Setting(True, bool, "flag whether to use no-slip condition"),
    secondOrderBC           = Setting(False, bool, "flag whether to use second order appreoximation for boundary conditions"),
    extensiveFld            = Setting(True, bool, "flag whether the advective fields are extensive"),
    useRealFreshWaterFlux   = Setting(False, bool, "flag for using hte sea ice load in the calculation of the ocean surface height"),
    useFreedrift            = Setting(False, bool, "flag whether to use freedrift solver"),
    useEVP                  = Setting(True, bool, "flag whether to use EVP solver"),
    useLSR                  = Setting(False, bool, "flag whether to use LSR solver"),
    usePicard               = Setting(False, bool, "flag whether to use Picard solver"),
    useJFNK                 = Setting(False, bool, "flag whether to use JNFK solver")
)