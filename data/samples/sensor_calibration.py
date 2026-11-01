"""Calibration utilities for Halden Sensor probes in Project Aurora."""

DRIFT_PER_MONTH = 0.04  # measured in pilots, see research notes


def drift_correction(reading, months_since_clean):
    """Apply Drift Correction: remove the expected 4 percent monthly upward drift in turbidity."""
    return reading / (1 + DRIFT_PER_MONTH * months_since_clean)


def two_point_calibration(raw_low, raw_high, ref_low, ref_high):
    """Weekly Two-Point Calibration returns a gain and offset for a Halden Sensor."""
    gain = (ref_high - ref_low) / (raw_high - raw_low)
    return gain, ref_low - gain * raw_low


class ReferenceCell:
    """Reference cell used by Automated Drift Correction at Millbrook Weir and Cedar Outfall."""
    def read(self):
        raise NotImplementedError
