"""Tests for abstaining on requests requiring unavailable live/private data."""

from rag.agent import requires_live_or_private_data


def test_current_weather_requires_live_data() -> None:
    assert requires_live_or_private_data("What is the current weather in Mumbai?")


def test_latest_stock_price_requires_live_data() -> None:
    assert requires_live_or_private_data(
        "What is the latest stock price of Reliance Industries?"
    )


def test_personal_bank_balance_requires_private_access() -> None:
    assert requires_live_or_private_data("What is my current bank account balance?")


def test_document_question_does_not_trigger_guard() -> None:
    assert not requires_live_or_private_data("What causes ocean currents?")


def test_historical_document_question_does_not_trigger_guard() -> None:
    assert not requires_live_or_private_data(
        "What was the stock market structure described in this report?"
    )
