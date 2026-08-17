"""Conservative waveform augmentation for dysarthric ASR experiments.

These transforms alter recording conditions and timing; they do not attempt to
synthesize a disability.  Apply them only to training examples, never dev/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    speed_probability: float = 0.5
    speed_min: float = 0.90
    speed_max: float = 1.10
    pause_probability: float = 0.5
    pause_min_seconds: float = 0.20
    pause_max_seconds: float = 1.20
    pause_repetitions_max: int = 1
    noise_probability: float = 0.35
    noise_snr_db_min: float = 18.0
    noise_snr_db_max: float = 35.0
    gain_probability: float = 0.5
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Classic resampling speed perturbation; factor > 1 produces longer audio."""
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    output_length = max(1, round(len(audio) * factor))
    source_positions = np.linspace(0, len(audio) - 1, output_length)
    return np.interp(source_positions, np.arange(len(audio)), audio).astype(np.float32)


def insert_pause_at_low_energy(
    audio: np.ndarray, sample_rate: int, seconds: float, rng: np.random.Generator
) -> np.ndarray:
    """Lengthen a naturally quiet internal boundary instead of cutting a phone."""
    frame = max(1, round(sample_rate * 0.02))
    margin = round(sample_rate * 0.25)
    if len(audio) < 2 * margin + frame:
        return audio
    starts = np.arange(0, len(audio) - frame + 1, frame)
    energies = np.array(
        [np.sqrt(np.mean(np.square(audio[start : start + frame]), dtype=np.float64)) for start in starts]
    )
    noise_floor = float(np.quantile(energies, 0.20))
    speech_level = float(np.quantile(energies, 0.95))
    if speech_level <= noise_floor * 1.05:
        # Nearly constant synthetic/test signals can have a short valid gap
        # below the 20th percentile without separating the two quantiles.
        threshold = max(speech_level * 0.5, 1e-7)
    else:
        threshold = max(noise_floor + 0.10 * (speech_level - noise_floor), 1e-7)
    active = np.flatnonzero(energies > threshold)
    if len(active) < 2:
        return audio
    internal = (starts >= max(margin, starts[active[0]] + frame)) & (
        starts <= min(len(audio) - margin - frame, starts[active[-1]] - frame)
    )
    choices = starts[internal & (energies <= threshold)]
    if not len(choices):
        return audio
    location = int(rng.choice(choices))
    pause_length = max(1, round(seconds * sample_rate))
    # A tiny measured-noise floor is safer than an impossible run of exact zeros.
    quiet_rms = float(max(np.median(energies[energies <= threshold]), 1e-7))
    pause = rng.normal(0.0, quiet_rms, pause_length).astype(np.float32)
    return np.concatenate((audio[:location], pause, audio[location:]))


def add_noise_at_snr(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if signal_rms == 0:
        return audio
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = rng.normal(0.0, noise_rms, len(audio)).astype(np.float32)
    return audio + noise


def augment(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    config: AugmentConfig = AugmentConfig(),
) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32)
    if result.ndim != 1:
        raise ValueError("audio must be mono")
    if rng.random() < config.speed_probability:
        result = speed_perturb(result, rng.uniform(config.speed_min, config.speed_max))
    if rng.random() < config.pause_probability:
        if config.pause_repetitions_max < 1:
            raise ValueError("pause_repetitions_max must be at least one")
        repetitions = int(rng.integers(1, config.pause_repetitions_max + 1))
        for _ in range(repetitions):
            seconds = rng.uniform(config.pause_min_seconds, config.pause_max_seconds)
            result = insert_pause_at_low_energy(result, sample_rate, seconds, rng)
    if rng.random() < config.noise_probability:
        result = add_noise_at_snr(
            result, rng.uniform(config.noise_snr_db_min, config.noise_snr_db_max), rng
        )
    if rng.random() < config.gain_probability:
        result = result * (10 ** (rng.uniform(config.gain_db_min, config.gain_db_max) / 20))
    return np.clip(result, -1.0, 1.0).astype(np.float32)
