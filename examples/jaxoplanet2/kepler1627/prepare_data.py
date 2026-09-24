"""Download Kepler-1627 (KIC 6184894) long-cadence photometry into kepler.csv.

Needs lightkurve (``pip install lightkurve``) and network access. Mirrors the
data preparation of docs/tutorials/lc-gp-transit_using_jaxoplanet.ipynb:
stitch all quarters, drop NaNs, quality-flagged cadences and upward outliers.
The transit and the starspot variability are then fitted jointly (no
detrending), exactly as in the notebook.
"""

from pathlib import Path

import lightkurve as lk
import numpy as np

HERE = Path(__file__).parent
BKJD_OFFSET = 2454833.0  # Kepler BJD - 2454833


def main() -> None:
    search = lk.search_lightcurve(
        "KIC 6184894", mission="Kepler", author="Kepler", cadence="long"
    )
    lc = search.download_all().stitch().remove_nans()
    lc = lc.remove_outliers(sigma_upper=4, sigma_lower=20)  # keep the transits
    lc = lc[lc.quality == 0]
    time = np.asarray(lc.time.value, dtype=float) + BKJD_OFFSET
    flux = np.asarray(lc.flux.value, dtype=float)
    err = np.asarray(lc.flux_err.value, dtype=float)
    order = np.argsort(time)
    rows = np.column_stack([time, flux, err])[order]
    np.savetxt(HERE / "kepler.csv", rows, delimiter=",", fmt="%.10f",
               header="time,flux,flux_err")  # fmt: skip
    print(f"wrote {len(rows)} points to {HERE / 'kepler.csv'}")


if __name__ == "__main__":
    main()
