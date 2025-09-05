from __future__ import annotations
import numpy as np

def snr_to_rate_mbps(bw_mhz: float, snr_linear: float) -> float:
    # Shannon-like proxy: R = BW * log2(1 + SNR)  (MHz -> Mbps)
    return float(bw_mhz * np.log2(1.0 + snr_linear))

def distance_to_snr(distance: float, tx_dbm: float, noise_dbm: float) -> float:
    # Simplified path-loss: Pr(dBm) = Pt - 20*log10(d) - C
    # C folds constants (antenna gains, etc.). Avoid d=0.
    d = max(distance, 1.0)
    c = 40.0
    pr_dbm = tx_dbm - (20.0 * np.log10(d)) - c
    snr_db = pr_dbm - noise_dbm
    return 10 ** (snr_db / 10.0)

def tx_time_seconds(bits: float, rate_mbps: float) -> float:
    rate_bps = max(rate_mbps, 1e-6) * 1e6
    return float(bits / rate_bps)

def compute_time_seconds(cycles: float, mips: float) -> float:
    rate = max(mips, 1e-6) * 1e6
    return float(cycles / rate)

def tx_energy_joules(bits: float, tx_dbm: float, tx_time_s: float, coeff_tx: float) -> float:
    # Scaled proxy: energy ∝ |P(dBm)| * t * bits_scale
    return float(coeff_tx * abs(tx_dbm) * tx_time_s * (bits / 1e6))

def compute_energy_joules(cycles: float, time_s: float, coeff_compute: float) -> float:
    return float(coeff_compute * cycles * time_s)
