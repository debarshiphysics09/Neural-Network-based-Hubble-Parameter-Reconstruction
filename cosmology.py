"""
Cosmological models used as ground truth for the synthetic benchmark.

Three fiducial models are implemented, spanning the realistic range of
background expansion histories consistent with current data:

  1. Flat LCDM           (Planck-like:    Om=0.30, Ok=0,    w0=-1, wa=0)
  2. Non-flat LCDM        (mildly open:    Om=0.30, Ok=0.05, w0=-1, wa=0)
  3. CPL dynamical DE     (w0waCDM:        Om=0.30, Ok=0,    w0=-0.9, wa=0.3)

All three share H0 = 70 km/s/Mpc so that reconstruction methods are
compared on equal footing; only the shape of H(z) differs.

c is in km/s.
"""
import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass

C_LIGHT = 299792.458  # km/s


@dataclass(frozen=True)
class Cosmology:
    name: str
    H0: float
    Om: float
    Ok: float = 0.0
    w0: float = -1.0
    wa: float = 0.0

    @property
    def Ode(self):
        return 1.0 - self.Om - self.Ok

    def de_density(self, z):
        """Dark-energy density evolution factor rho_DE(z)/rho_DE(0)."""
        a = 1.0 / (1.0 + z)
        if self.w0 == -1.0 and self.wa == 0.0:
            return np.ones_like(np.atleast_1d(z), dtype=float)
        # CPL parametrisation: w(a) = w0 + wa (1-a)
        return a ** (-3 * (1 + self.w0 + self.wa)) * np.exp(-3 * self.wa * (1 - a))

    def Hz(self, z):
        """Hubble parameter H(z) in km/s/Mpc."""
        z = np.atleast_1d(z).astype(float)
        E2 = (self.Om * (1 + z) ** 3
              + self.Ok * (1 + z) ** 2
              + self.Ode * self.de_density(z))
        return self.H0 * np.sqrt(E2)

    def q(self, z):
        """
        Deceleration parameter q(z) = -1 - dlnH/dln(1+z) ... derived
        analytically via q(z) = (1+z)/H(z) * dH/dz - 1, computed with a
        high-accuracy numerical derivative of the exact H(z) above.
        """
        z = np.atleast_1d(z).astype(float)
        h = 1e-5
        dHdz = (self.Hz(z + h) - self.Hz(z - h)) / (2 * h)
        Hz = self.Hz(z)
        return (1 + z) * dHdz / Hz - 1

    def transition_redshift(self, zmax=3.0, n=20000):
        """Locate the redshift where q(z) changes sign (deceleration -> acceleration)."""
        zgrid = np.linspace(0, zmax, n)
        qgrid = self.q(zgrid)
        sign_change = np.where(np.diff(np.sign(qgrid)) != 0)[0]
        if len(sign_change) == 0:
            return np.nan
        i = sign_change[0]
        # linear interpolation for sub-grid precision
        z1, z2 = zgrid[i], zgrid[i + 1]
        q1, q2 = qgrid[i], qgrid[i + 1]
        return z1 - q1 * (z2 - z1) / (q2 - q1)

    def comoving_distance(self, z):
        """D_C(z) in Mpc, for a scalar or array z (uses quad per element)."""
        z = np.atleast_1d(z).astype(float)
        out = np.empty_like(z)
        for i, zi in enumerate(z):
            integrand = lambda zp: C_LIGHT / self.Hz(zp)[0]
            out[i] = quad(integrand, 0, zi, limit=200)[0]
        return out

    def luminosity_distance(self, z):
        z = np.atleast_1d(z).astype(float)
        Dc = self.comoving_distance(z)
        if self.Ok == 0:
            Dm = Dc
        else:
            DH0 = C_LIGHT / self.H0
            sqrtOk = np.sqrt(abs(self.Ok))
            x = sqrtOk * Dc / DH0
            if self.Ok > 0:
                Dm = DH0 / sqrtOk * np.sinh(x)
            else:
                Dm = DH0 / sqrtOk * np.sin(x)
        return (1 + z) * Dm

    def distance_modulus(self, z):
        DL = self.luminosity_distance(z)  # Mpc
        return 5 * np.log10(DL) + 25

    def D_H(self, z):
        """Hubble distance D_H(z) = c / H(z), the BAO line-of-sight observable proxy."""
        return C_LIGHT / self.Hz(z)


# ─── Fiducial models used throughout the benchmark ──────────────────────────
FLAT_LCDM    = Cosmology(name="Flat LCDM",      H0=70.0, Om=0.30, Ok=0.00, w0=-1.0,  wa=0.0)
NONFLAT_LCDM = Cosmology(name="Non-flat LCDM",  H0=70.0, Om=0.30, Ok=0.05, w0=-1.0,  wa=0.0)
CPL_DE       = Cosmology(name="CPL dynamical DE", H0=70.0, Om=0.30, Ok=0.00, w0=-0.9, wa=0.3)

ALL_COSMOLOGIES = {
    "flat_lcdm": FLAT_LCDM,
    "nonflat_lcdm": NONFLAT_LCDM,
    "cpl_de": CPL_DE,
}

if __name__ == "__main__":
    for key, cosmo in ALL_COSMOLOGIES.items():
        zt = cosmo.transition_redshift()
        print(f"{cosmo.name:18s}  H(1)={cosmo.Hz(1.0)[0]:7.3f}  z_t={zt:.4f}")
