"""
Real cosmic-chronometer (CC) H(z) compilation.

This is the standard 31-point differential-age compilation assembled from
Jimenez et al. (2003), Simon, Verde & Jimenez (2005), Stern et al. (2010),
Moresco et al. (2012), Zhang et al. (2014), Moresco (2015), Moresco et al.
(2016), and Ratsimbazafy et al. (2017), covering 0.07 < z < 1.965. This
compilation (or the near-identical 32-point version with the Borghi et al.
2022 z~0.75 point swapped in) is the one used throughout the CC/H(z)
reconstruction literature; see Moresco et al. (2022, Living Rev. Relativ.,
25, 6) for the up-to-date review and covariance treatment.

Columns: z, H(z) [km/s/Mpc], sigma_H [km/s/Mpc].

NOTE ON SCOPE: this dataset carries partially correlated systematic
uncertainties (shared stellar-population-synthesis modelling across
points) that are NOT included here -- we use only the quoted diagonal
statistical+systematic sigma_i, consistent with the idealised-noise
scope stated in the paper (Sec. 5.3). A covariance-matrix treatment
(Moresco et al. 2020, 2022) is the natural next step and is noted
explicitly as a limitation of this real-data application.
"""
import numpy as np

# z, H(z), sigma_H
_CC_TABLE = [
    (0.070, 69.0, 19.6),
    (0.090, 69.0, 12.0),
    (0.120, 68.6, 26.2),
    (0.170, 83.0, 8.0),
    (0.1791, 75.0, 4.0),
    (0.1993, 75.0, 5.0),
    (0.200, 72.9, 29.6),
    (0.270, 77.0, 14.0),
    (0.280, 88.8, 36.6),
    (0.3519, 83.0, 14.0),
    (0.3802, 83.0, 13.5),
    (0.400, 95.0, 17.0),
    (0.4004, 77.0, 10.2),
    (0.4247, 87.1, 11.2),
    (0.4497, 92.8, 12.9),
    (0.470, 89.0, 34.0),
    (0.4783, 80.9, 9.0),
    (0.480, 97.0, 62.0),
    (0.5929, 104.0, 13.0),
    (0.6797, 92.0, 8.0),
    (0.7812, 105.0, 12.0),
    (0.8754, 125.0, 17.0),
    (0.880, 90.0, 40.0),
    (0.900, 117.0, 23.0),
    (1.037, 154.0, 20.0),
    (1.300, 168.0, 17.0),
    (1.363, 160.0, 33.6),
    (1.430, 177.0, 18.0),
    (1.530, 140.0, 14.0),
    (1.750, 202.0, 40.0),
    (1.965, 186.5, 50.4),
]


def load_cc_data():
    """Return (z, H_obs, sigma_H) sorted by redshift, as float64 arrays."""
    arr = np.array(_CC_TABLE, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1], arr[:, 2]


if __name__ == "__main__":
    z, H, sig = load_cc_data()
    print(f"N = {len(z)} points, z in [{z.min():.3f}, {z.max():.3f}]")
