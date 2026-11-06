"""Detector registry. Add a detector here and every consumer picks it up."""

from __future__ import annotations

from faultline.detectors.base import Detector
from faultline.detectors.fit import DataFitDetector
from faultline.detectors.leakage import LeakageDetector
from faultline.detectors.pipeline_vulns import PipelineVulnDetector
from faultline.discovery.sources import SourceFile
from faultline.schema import Finding

DETECTORS: tuple[Detector, ...] = (PipelineVulnDetector(), LeakageDetector(), DataFitDetector())


def run_detectors(src: SourceFile, detectors: tuple[Detector, ...] = DETECTORS) -> list[Finding]:
    return [f for d in detectors for f in d.detect(src)]
