"""Unit tests for TimingReport."""

import time
import pytest


def test_stage_records_elapsed_time():
    from ltx_npu.timing import TimingReport
    timer = TimingReport()
    with timer.stage("test_stage"):
        time.sleep(0.05)
    assert "test_stage" in timer.stages
    assert timer.stages["test_stage"] >= 0.04


def test_multiple_stages():
    from ltx_npu.timing import TimingReport
    timer = TimingReport()
    with timer.stage("a"):
        time.sleep(0.02)
    with timer.stage("b"):
        time.sleep(0.02)
    assert len(timer.stages) == 2
    assert "a" in timer.stages
    assert "b" in timer.stages


def test_total():
    from ltx_npu.timing import TimingReport
    timer = TimingReport()
    with timer.stage("x"):
        time.sleep(0.02)
    with timer.stage("y"):
        time.sleep(0.02)
    total = timer.total()
    assert total >= 0.03


def test_report_format():
    from ltx_npu.timing import TimingReport
    timer = TimingReport()
    with timer.stage("Stage 1"):
        pass
    report = timer.report()
    assert "Stage 1" in report
    assert "Time" in report or "time" in report.lower()


def test_empty_report():
    from ltx_npu.timing import TimingReport
    timer = TimingReport()
    report = timer.report()
    assert isinstance(report, str)
