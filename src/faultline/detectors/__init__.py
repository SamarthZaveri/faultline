"""Detector registry. Add a detector here and every consumer picks it up."""

from __future__ import annotations

from collections.abc import Sequence

from faultline.contract import DatasetProfile
from faultline.detectors.base import Detector
from faultline.detectors.fit import DataFitDetector
from faultline.detectors.leakage import LeakageDetector
from faultline.detectors.pipeline_vulns import PipelineVulnDetector
from faultline.discovery.sources import SourceFile
from faultline.schema import Finding

DETECTORS: tuple[Detector, ...] = (PipelineVulnDetector(), LeakageDetector(), DataFitDetector())


def with_profiles(
    detectors: tuple[Detector, ...], profiles: Sequence[DatasetProfile]
) -> tuple[Detector, ...]:
    """The same detectors, told what the data looks like (data-fit rules; entity keys for leakage)."""
    keys = {k for p in profiles for k in p.group_keys}
    out: list[Detector] = []
    for d in detectors:
        if isinstance(d, DataFitDetector):
            out.append(DataFitDetector(profiles))
        elif isinstance(d, LeakageDetector):
            out.append(LeakageDetector(group_keys=keys))
        else:
            out.append(d)
    return tuple(out)


def run_detectors(src: SourceFile, detectors: tuple[Detector, ...] = DETECTORS) -> list[Finding]:
    return [f for d in detectors for f in d.detect(src)]
