"""
Real line-of-sight BAO H(z) compilation.

These are direct H(z) values inferred from published radial/line-of-sight
BAO measurements (as opposed to the transverse D_M(z) or the isotropic
D_V(z), which do not give H(z) directly). This is the standard set used
throughout the GP/H(z)-reconstruction literature to supplement the CC
compilation (e.g. Sharov & Vasiliev 2018; Yu, Ratra & Wang 2018; Gomez-
Valent & Sola Peracaula 2018, and many others use this or a near-identical
set), drawing on:

  - SDSS DR7 LRG:         Gaztanaga, Cabre & Hui (2009); Oka et al. (2014)
  - SDSS DR7 (reanalysis): Chuang & Wang (2013)
  - BOSS DR9 CMASS:       Anderson et al. (2014); Samushia et al. (2013)
  - WiggleZ:              Blake et al. (2012)
  - BOSS DR11 Lya-forest:  Busca et al. (2013); Delubac et al. (2015);
                           Font-Ribera et al. (2014, Lya-QSO cross-correlation)

CAUTION (stated explicitly, not hidden): these numbers are reproduced from
memory of the widely-circulated secondary compilation used across dozens
of papers, not re-transcribed from the primary tables in this session
(no network access to the original journals here). Before this goes into
any real submission, EVERY value below must be re-verified line-by-line
against the original source papers listed. This file is clearly marked
as provisional for exactly that reason.

REFEREE FIX (independence pruning): an earlier version of this table
double-counted three effectively non-independent measurements: (i) two
adjacent points both attributed to Chuang & Wang (2013) at z=0.34/0.35
(a single analysis, listed twice by transcription error); (ii) two BOSS
DR9 CMASS points at z=0.57 from Anderson (2014) and Samushia (2013) --
the same survey/sample analysed twice, not independent measurements;
(iii) both Busca (2013) and Delubac (2015) at the BOSS DR11 Lya-forest
autocorrelation redshift, where Delubac (2015) is an updated analysis of
the SAME sample that supersedes Busca (2013). All three redundancies are
now removed, keeping one point per independent survey/analysis. The
Font-Ribera (2014) Lya-QSO CROSS-correlation point is retained alongside
Delubac (2015)'s AUTO-correlation point: both use the BOSS DR11 Lya
dataset, so they are not fully independent either (shared spectra), but
they use different estimators/statistics and are conventionally treated
as complementary rather than redundant in the literature; this residual
non-independence is disclosed in the manuscript rather than pruned
further, since removing it would leave only one high-z anchor point.

REFEREE CAVEAT (physical, not just statistical): these are not
model-independent H(z) values in the same sense as the CC points. Radial
BAO measurements constrain H(z) r_d (r_d = the sound horizon at the drag
epoch); converting to H(z) in km/s/Mpc requires dividing by an assumed
fiducial r_d, which is itself computed from early-universe physics in a
specific cosmology (typically a Planck-calibrated value). The BAO points
below therefore carry a soft, implicit model dependence that the CC
points do not. This is stated explicitly in the manuscript wherever BAO
data are used, and readers should not treat "real BAO data" as being on
equal non-parametric footing with "real CC data."

REFEREE CAVEAT (WiggleZ covariance): the four WiggleZ points (Blake et
al. 2012) at z=0.43, 0.44, 0.60, 0.73 come from a single survey and are
published with a non-trivial covariance matrix in the original paper;
we use only their diagonal (quoted marginal) uncertainties here, which
understates their true joint uncertainty. This is a real, disclosed
limitation, not an oversight.

Columns: z, H(z) [km/s/Mpc], sigma_H [km/s/Mpc], source.
"""
import numpy as np

_BAO_TABLE = [
    # z,     H(z),  sigma,  source
    (0.24, 79.69, 2.65,  "Gaztanaga+2009 (SDSS DR7 LRG)"),
    (0.30, 81.7,  6.22,  "Oka+2014 (SDSS DR7)"),
    (0.35, 82.7,  8.4,   "Chuang&Wang2013 (SDSS DR7) [merged duplicate z=0.34/0.35 entry]"),
    (0.43, 86.45, 3.68,  "Blake+2012 (WiggleZ) [diagonal sigma only, see caveat]"),
    (0.44, 82.6,  7.8,   "Blake+2012 (WiggleZ) [diagonal sigma only, see caveat]"),
    (0.57, 92.4,  4.5,   "Anderson+2014 (BOSS DR9 CMASS) [Samushia+2013 duplicate removed]"),
    (0.60, 87.9,  6.1,   "Blake+2012 (WiggleZ) [diagonal sigma only, see caveat]"),
    (0.73, 97.3,  7.0,   "Blake+2012 (WiggleZ) [diagonal sigma only, see caveat]"),
    (2.33, 224.0, 8.0,   "Delubac+2015 (BOSS DR11 Lya forest) [supersedes Busca+2013, removed]"),
    (2.36, 226.0, 8.0,   "Font-Ribera+2014 (BOSS Lya-QSO cross-correlation)"),
]


def load_bao_data():
    arr_num = np.array([(r[0], r[1], r[2]) for r in _BAO_TABLE], dtype=float)
    order = np.argsort(arr_num[:, 0])
    sources = [ _BAO_TABLE[i][3] for i in order ]
    arr_num = arr_num[order]
    return arr_num[:, 0], arr_num[:, 1], arr_num[:, 2], sources


if __name__ == "__main__":
    z, H, sig, src = load_bao_data()
    for zi, Hi, si, s in zip(z, H, sig, src):
        print(f"z={zi:.2f}  H={Hi:6.1f} +/- {si:4.1f}   {s}")
