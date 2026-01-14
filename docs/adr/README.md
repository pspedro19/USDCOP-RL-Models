# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the USDCOP Trading System.

## What is an ADR?

An Architecture Decision Record (ADR) documents significant architectural decisions along with their context and consequences. ADRs help:

- Record decisions for future reference
- Understand why things are the way they are
- Onboard new team members
- Avoid re-litigating past decisions

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](ADR-0001-wilder-ema-for-technical-indicators.md) | Use Wilder's EMA for RSI, ATR, and ADX | Accepted | 2025-01-14 |
| [ADR-0002](ADR-0002-feature-circuit-breaker.md) | Feature Circuit Breaker for Data Quality | Accepted | 2025-01-14 |

## Creating a New ADR

1. Copy `TEMPLATE.md` to `ADR-XXXX-descriptive-title.md`
2. Fill in all sections
3. Submit for review
4. Update this index when accepted

## Status Definitions

- **Proposed**: Under discussion
- **Accepted**: Approved and implemented
- **Deprecated**: No longer relevant but kept for history
- **Superseded**: Replaced by another ADR

## Template

See [TEMPLATE.md](TEMPLATE.md) for the standard ADR format.
