from __future__ import annotations

import math
import pandas as pd
import qc_config_v2 as cfg
import execution_v2 as ex


def _position_snapshot(algorithm, structure):
    """Return executable liquidation P&L plus midpoint diagnostics.

    ``gross`` is intentionally the NATURAL executable P&L and therefore is the
    number consumed by the unchanged PT/SL rules in manage_open_positions.
    """
    now = pd.Timestamp(algorithm.time).normalize()
    remaining = int((structure['expiry'] - now).days)

    symbols = list(structure.get('symbols', []))
    qtys = [int(q) for q in structure.get('qtys', [])]
    if not symbols or len(symbols) != len(qtys):
        return None

    mid_mark = ex._signed_mark(algorithm, symbols, qtys)
    close_qtys = [-q for q in qtys]
    close_natural = ex._signed_natural(algorithm, symbols, close_qtys)
    if mid_mark is None or close_natural is None:
        return None

    contracts = int(structure.get('contracts', 0) or 0)
    if contracts <= 0:
        return None

    premium_type = str(structure.get('premium_type', '')).upper()
    mult = float(cfg.CONTRACT_MULTIPLIER)

    if premium_type == 'CREDIT':
        credit = float(structure.get('entry_credit', 0.0) or 0.0)
        entry_signed = -credit
        gross_mid = (float(mid_mark) + credit) * mult * contracts
        base = max(credit * mult * contracts, 1e-9)
    else:
        debit = float(structure.get('entry_debit', 0.0) or 0.0)
        entry_signed = debit
        gross_mid = (float(mid_mark) - debit) * mult * contracts
        base = max(debit * mult * contracts, 1e-9)

    # Cash P&L = -(signed entry package + signed natural close package).
    gross_exec = -(entry_signed + float(close_natural)) * mult * contracts

    return {
        'remaining': remaining,
        'mark': float(mid_mark),
        'close_natural': float(close_natural),
        'gross_mid': float(gross_mid),
        'gross': float(gross_exec),
        'base': float(base),
        'profit_target': float(cfg.PROFIT_TARGET_FRACTION * base),
        'stop_loss': float(-cfg.STOP_LOSS_MULTIPLE * base),
    }


def _registry_key_for_structure(algorithm, structure):
    for key, value in getattr(algorithm, '_vh_open_structures', {}).items():
        if value is structure:
            return key
    return None


def _schedule_flatten_verify(algorithm, structure, attempt, delay_minutes=2):
    key = _registry_key_for_structure(algorithm, structure)
    if key is None:
        return False

    when = pd.Timestamp(algorithm.time) + pd.Timedelta(minutes=int(delay_minutes))
    if when.date() != pd.Timestamp(algorithm.time).date():
        return False

    try:
        dr = algorithm.date_rules.on(int(when.year), int(when.month), int(when.day))
        tr = algorithm.time_rules.at(int(when.hour), int(when.minute))

        def _callback():
            _verify_individual_flatten(algorithm, key, int(attempt))

        algorithm.schedule.on(dr, tr, _callback)
        return True
    except Exception as exc:
        algorithm.error(
            f"INDIVIDUAL EXIT VERIFY SCHEDULE FAILED {key}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def _verify_individual_flatten(algorithm, key, attempt):
    ex._prune_registry(algorithm)
    structure = getattr(algorithm, '_vh_open_structures', {}).get(key)
    if structure is None or structure.get('closed'):
        return

    if not ex._has_structure_holding(algorithm, structure):
        structure['closed'] = True
        structure['close_pending'] = False
        return

    # Do not permit any previous close ticket (including a stale combo-market
    # ticket) to survive while we flatten residual holdings individually.
    try:
        ex._cancel_close_tickets_intentionally(
            algorithm,
            structure,
            f"Individual market verify attempt {attempt}",
        )
    except Exception:
        pass

    tag = str(structure.get('close_tag', 'VH_EXIT'))
    tickets = ex._submit_individual_flatten(algorithm, structure, tag)
    structure['close_ticket_ids'] = ex._ticket_ids(tickets)
    structure['close_pending'] = bool(structure['close_ticket_ids'])
    structure['close_stage'] = 'INDIVIDUAL_MARKET'
    structure['close_submit_time'] = pd.Timestamp(algorithm.time)
    structure['force_individual_close'] = True

    if not ex._has_structure_holding(algorithm, structure):
        structure['closed'] = True
        structure['close_pending'] = False
        return

    if int(attempt) < 2:
        _schedule_flatten_verify(algorithm, structure, int(attempt) + 1, 2)
    else:
        algorithm.error(
            f"INDIVIDUAL EXIT RESIDUAL {structure.get('ticker')} "
            f"after {attempt} verification attempts"
        )


def _submit_market_fallback(algorithm, structure, tag):
    """Final fallback = individual-leg market flatten, never combo-market."""
    if not ex._has_structure_holding(algorithm, structure):
        return []

    # Defensive: cancel any working grouped close order first.
    try:
        ex._cancel_close_tickets_intentionally(
            algorithm,
            structure,
            'Final fallback -> individual market',
        )
    except Exception:
        pass

    structure['force_individual_close'] = True
    ticket_list = ex._submit_individual_flatten(algorithm, structure, tag)

    structure['close_ticket_ids'] = ex._ticket_ids(ticket_list)
    structure['close_pending'] = bool(structure['close_ticket_ids'])
    structure['close_tag'] = tag
    structure['close_stage'] = 'INDIVIDUAL_MARKET'
    structure['close_submit_time'] = pd.Timestamp(algorithm.time)
    structure['close_failed'] = False

    algorithm.debug(
        f"EXIT INDIVIDUAL MARKET FALLBACK {structure.get('ticker')} "
        f"{structure.get('structure')} {tag}"
    )

    if not ex._has_structure_holding(algorithm, structure):
        structure['closed'] = True
        structure['close_pending'] = False
        return ticket_list

    _schedule_flatten_verify(algorithm, structure, 1, 2)
    return ticket_list


# Exit-only V3.3 overlay: executable-natural management and individual-market fallback.
ex._position_snapshot = _position_snapshot
ex._submit_market_fallback = _submit_market_fallback


