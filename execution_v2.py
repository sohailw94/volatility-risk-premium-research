import math
import pandas as pd
from AlgorithmImports import Leg, OrderStatus
import qc_config_v2 as cfg
_ACTIVE_STATUSES = {OrderStatus.NEW, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
_FAILED_STATUSES = {OrderStatus.INVALID, OrderStatus.CANCELED}

def _signed_mark(algorithm, symbols, qtys):
  total = 0.0
  for symbol, qty in zip(symbols, qtys):
    sec = algorithm.securities[symbol]
    bid = float(sec.bid_price or 0.0)
    ask = float(sec.ask_price or 0.0)
    if ask <= 0.0 or bid < 0.0 or ask < bid:
      return None
    total += int(qty) * 0.5 * (bid + ask)
  return float(total)

def _signed_natural(algorithm, symbols, qtys):
  total = 0.0
  for symbol, qty in zip(symbols, qtys):
    sec = algorithm.securities[symbol]
    bid = float(sec.bid_price or 0.0)
    ask = float(sec.ask_price or 0.0)
    if ask <= 0.0 or bid < 0.0 or ask < bid:
      return None
    qty = int(qty)
    px = ask if qty > 0 else bid
    total += qty * px
  return float(total)

def _get_ticket(algorithm, ticket_id):
  try:
    return algorithm.transactions.get_order_ticket(int(ticket_id))
  except Exception:
    return None

def _ticket_open(algorithm, ticket_id):
  ticket = _get_ticket(algorithm, ticket_id)
  return ticket is not None and ticket.status in _ACTIVE_STATUSES

def _ticket_ids(value):
  if value is None:
    return []
  try:
    return [int(ticket.order_id) for ticket in list(value) if ticket is not None]
  except Exception:
    try:
      return [int(value.order_id)]
    except Exception:
      return []

def _holding_qty(algorithm, symbol):
  try:
    return int(algorithm.portfolio[symbol].quantity)
  except Exception:
    return 0

def _has_structure_holding(algorithm, structure):
  return any((_holding_qty(algorithm, s) != 0 for s in structure.get('symbols', [])))

def _has_open_order_for_symbol(algorithm, symbol):
  try:
    for ticket in algorithm.transactions.get_open_order_tickets():
      try:
        order = ticket.get_order_by_id(ticket.order_id)
        if order is not None and order.symbol == symbol:
          return True
      except Exception:
        try:
          order = algorithm.transactions.get_order_by_id(ticket.order_id)
          if order is not None and order.symbol == symbol:
            return True
        except Exception:
          pass
  except Exception:
    pass
  return False

def _prune_registry(algorithm):
  registry = getattr(algorithm, '_vh_open_structures', {})
  for structure in registry.values():
    if structure.get('closed'):
      continue
    has_holding = _has_structure_holding(algorithm, structure)
    entry_open = any((_ticket_open(algorithm, i) for i in structure.get('entry_ticket_ids', [])))
    close_open = any((_ticket_open(algorithm, i) for i in structure.get('close_ticket_ids', [])))
    if structure.get('close_pending') and (not close_open):
      if not has_holding:
        structure['closed'] = True
        structure['close_pending'] = False
        if not structure.get('close_confirmed_logged'):
          structure['close_confirmed_logged'] = True
      else:
        structure['close_pending'] = False
    if not structure.get('close_pending') and (not entry_open) and (not has_holding):
      structure['closed'] = True

def _existing_ticker(algorithm, ticker):
  _prune_registry(algorithm)
  for structure in getattr(algorithm, '_vh_open_structures', {}).values():
    if not structure.get('closed') and structure.get('ticker') == ticker:
      return True
  return False

def _risk_per_contract(row):
  value = float(row.get('max_loss', 0.0) or 0.0)
  return max(value * cfg.CONTRACT_MULTIPLIER, 0.0)

def _risk_caps(algorithm, row, requested):
  equity = max(float(algorithm.portfolio.total_portfolio_value), 1.0)
  rpc = _risk_per_contract(row)
  if rpc <= 0.0:
    return 0
  existing_total = 0.0
  existing_ticker = 0.0
  ticker = str(row.ticker)
  for structure in getattr(algorithm, '_vh_open_structures', {}).values():
    if structure.get('closed'):
      continue
    risk_dollars = float(structure.get('risk_dollars', 0.0) or 0.0)
    existing_total += risk_dollars
    if structure.get('ticker') == ticker:
      existing_ticker += risk_dollars
  uncapped=str(row.get('risk_class','CAPPED')).upper()=='UNCAPPED'
  default_single=float(getattr(cfg,'UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT',cfg.MAX_SINGLE_TRADE_RISK_PCT)) if uncapped else float(cfg.MAX_SINGLE_TRADE_RISK_PCT)
  default_ticker=float(getattr(cfg,'UNCAPPED_MAX_TICKER_RISK_PCT',cfg.MAX_TICKER_RISK_PCT)) if uncapped else float(cfg.MAX_TICKER_RISK_PCT)
  single_pct=float(row.get('hjb_max_single_risk_pct',default_single) or default_single)
  ticker_pct=float(row.get('hjb_max_ticker_risk_pct',default_ticker) or default_ticker)
  single = int(math.floor(equity * single_pct / 100.0 / rpc))
  ticker_cap = int(math.floor(max(equity * ticker_pct / 100.0 - existing_ticker, 0.0) / rpc))
  total = int(math.floor(max(equity * cfg.MAX_TOTAL_DEFINED_RISK_PCT / 100.0 - existing_total, 0.0) / rpc))
  return max(min(int(requested), single, ticker_cap, total), 0)

def _entry_limit_exec(row, algorithm):
  syms = list(row.lean_leg_symbols)
  qtys = [int(x) for x in row.lean_signed_qty]
  cap = row.get('liquidity_spread_cap', None)
  try:
    cap = float(cap)
  except Exception:
    cap = None
  for sym in syms:
    sec = algorithm.securities[sym]
    bid = float(sec.bid_price or 0.0)
    ask = float(sec.ask_price or 0.0)
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
      return None
    mid = 0.5 * (bid + ask)
    if cap is not None and math.isfinite(cap) and mid>0 and (ask-bid)/mid>cap and (ask-bid)>float(getattr(cfg,'LIQUIDITY_ABSOLUTE_SPREAD_OVERRIDE',0.03)):
      return None
  mid = _signed_mark(algorithm, syms, qtys)
  nat = _signed_natural(algorithm, syms, qtys)
  if mid is None or nat is None:
    return None
  mode = str(getattr(cfg, 'ENTRY_EXECUTION_MODE', 'NATURAL_MARKETABLE')).upper()
  pad = max(float(getattr(cfg, 'ENTRY_MARKETABLE_CUSHION', 0.01)), 0.0)
  if str(row.premium_type).upper() == 'CREDIT':
    mc = -float(mid)
    nc = -float(nat)
    if mc <= 0 or nc <= 0:
      return None
    floor = max(mc * float(cfg.MIN_CREDIT_FRACTION_OF_MID), float(cfg.MIN_NET_CREDIT))
    if mode == 'NATURAL_MARKETABLE':
      if nc < floor:
        return None
      return float(nat) + pad
    return -floor
  if float(mid) <= 0 or float(nat) <= 0:
    return None
  return float(nat) + pad if mode == 'NATURAL_MARKETABLE' else max(float(mid), 0.01)

def _research_structure_code(name):
  mapping = {'IRON_CONDOR': 'IC', 'LONG_STRADDLE': 'LS', 'LONG_STRANGLE': 'LG', 'LONG_CALL_BUTTERFLY': 'BF', 'SHORT_PUT_VRP':'SPUT', 'SHORT_CALL_VRP':'SCAL', 'SHORT_STRANGLE_APPROVED':'SSTA'}
  return mapping.get(str(name).upper(), str(name).upper()[:4])

def _fmt_tag_number(value, decimals=0):
  try:
    x = float(value)
    if not math.isfinite(x):
      return 'NA'
    if decimals == 0:
      return str(int(round(x)))
    return f'{x:.{decimals}f}'
  except Exception:
    return 'NA'

def _research_entry_tag(algorithm, row, contracts):
  d = pd.Timestamp(algorithm.time).strftime('%y%m%d')
  ticker = str(row.get('ticker', '')).upper()
  structure = _research_structure_code(row.get('structure', ''))
  ev = _fmt_tag_number(row.get('mc_expected_pnl'))
  robust = _fmt_tag_number(row.get('mc_p25_scenario_expected_pnl'))
  worst = _fmt_tag_number(row.get('mc_worst_case_expected_pnl'))
  cvar = _fmt_tag_number(row.get('mc_cvar_95'))
  pop = _fmt_tag_number(row.get('mc_pop'), 2)
  action = _fmt_tag_number(row.get('hjb_action_size'), 2)
  risk = _fmt_tag_number(_risk_per_contract(row))
  tag = f'VH|D={d}|T={ticker}|S={structure}|N={int(contracts)}|EV={ev}|R={robust}|W={worst}|C={cvar}|P={pop}|A={action}|RK={risk}'
  return tag[:100]

def _register(algorithm, row, contracts, tickets=None):
  if not hasattr(algorithm, '_vh_open_structures'):
    algorithm._vh_open_structures = {}
  key = f'{row.ticker}|{pd.Timestamp(row.expiry).date()}|{row.structure}|{len(algorithm._vh_open_structures)}'
  algorithm._vh_open_structures[key] = {'ticker': str(row.ticker), 'structure': str(row.structure), 'signal_date': pd.Timestamp(algorithm.time).strftime('%y%m%d'), 'expiry': pd.Timestamp(row.expiry).normalize(), 'symbols': list(row.lean_leg_symbols), 'qtys': [int(x) for x in row.lean_signed_qty], 'contracts': int(contracts), 'premium_type': str(row.premium_type).upper(), 'risk_class': str(row.get('risk_class','CAPPED')).upper(), 'option_leg_role': str(row.get('option_leg_role','PACKAGE')).upper(), 'package_only_management': bool(row.get('package_only_management',False)), 'paired_independent_approval': bool(row.get('paired_independent_approval',False)), 'model_entry_credit': float(row.get('net_credit', 0.0) or 0.0), 'model_entry_debit': float(row.get('net_debit', 0.0) or 0.0), 'entry_credit': float(row.get('net_credit', 0.0) or 0.0), 'entry_debit': float(row.get('net_debit', 0.0) or 0.0), 'model_risk_per_contract': _risk_per_contract(row), 'risk_dollars': _risk_per_contract(row) * int(contracts), 'actual_entry_synced': False, 'actual_entry_package_signed': None, 'actual_entry_price_gap': None, 'entry_ticket_ids': _ticket_ids(tickets), 'close_ticket_ids': [], 'entry_failed': False, 'close_failed': False, 'force_individual_close': False, 'close_pending': False, 'close_tag': '', 'close_stage': None, 'close_cycle': 0, 'close_reason': None, 'close_submit_time': None, 'intentional_close_cancel_ids': [], 'closed': False}

def submit_hjb_combo_orders(algorithm, sized, execute_orders=True):
  sent = []
  day = str(algorithm.time.date())
  if getattr(algorithm, '_vh_entry_day', None) != day:
    algorithm._vh_entry_day = day
    algorithm._vh_entries_today = 0
  if sized is None or sized.empty:
    return sent
  for _, row in sized.iterrows():
    if getattr(algorithm, '_vh_entries_today', 0) >= cfg.MAX_NEW_TRADES_PER_DAY:
      break
    action = float(row.get('hjb_action_size', 0.0) or 0.0)
    if action <= 0.0:
      continue
    ticker = str(row.ticker)
    if _existing_ticker(algorithm, ticker):
      continue
    requested = int(row.get('hjb_contracts', 0) or 0)
    uncapped=str(row.get('risk_class','CAPPED')).upper()=='UNCAPPED'
    if requested <= 0:
      equity = max(float(algorithm.portfolio.total_portfolio_value), 1.0)
      rpc = _risk_per_contract(row)
      budget = equity * float(row.get('hjb_position_pct', 0.0) or 0.0) / 100.0
      requested = int(math.floor(budget / max(rpc, 1e-09)))
      if uncapped and action>0.0 and requested<=0:
        default_cap=float(getattr(cfg,'UNCAPPED_MAX_SINGLE_TRADE_RISK_PCT',cfg.MAX_SINGLE_TRADE_RISK_PCT))
        cap=equity*float(row.get('hjb_max_single_risk_pct',default_cap) or default_cap)/100.0
        requested=1 if rpc<=cap else 0
    default_max=int(getattr(cfg,'UNCAPPED_MAX_CONTRACTS_PER_TRADE',1)) if uncapped else int(cfg.MAX_CONTRACTS_PER_TRADE)
    max_contracts=int(row.get('max_contracts_per_trade',default_max) or default_max)
    requested = min(requested, max_contracts)
    contracts = _risk_caps(algorithm, row, requested)
    if contracts <= 0:
      continue
    symbols = list(row.lean_leg_symbols)
    qtys = [int(x) for x in row.lean_signed_qty]
    if len(symbols) != len(qtys) or not symbols:
      continue
    limit = _entry_limit_exec(row, algorithm)
    if limit is None:
      continue
    if not execute_orders:
      continue
    try:
      legs = [Leg.create(symbol, qty) for symbol, qty in zip(symbols, qtys)]
      tickets = algorithm.combo_limit_order(legs, contracts, float(round(limit, 2)), tag=_research_entry_tag(algorithm, row, contracts))
      ticket_list = list(tickets) if tickets is not None else []
      sent.extend(ticket_list)
      _register(algorithm, row, contracts, ticket_list)
      algorithm._vh_entries_today = getattr(algorithm, '_vh_entries_today', 0) + 1
    except Exception as exc:
      algorithm.error(f'ENTRY FAILED {ticker}: {type(exc).__name__}: {exc}')
  return sent

def _submit_individual_flatten(algorithm, structure, tag):
  tickets = []
  for symbol in structure.get('symbols', []):
    qty = _holding_qty(algorithm, symbol)
    if qty == 0 or _has_open_order_for_symbol(algorithm, symbol):
      continue
    try:
      ticket = algorithm.market_order(symbol, -qty, tag=tag)
      if ticket is not None:
        tickets.append(ticket)
    except Exception as exc:
      algorithm.error(f"INDIVIDUAL EXIT FAILED {structure['ticker']} {symbol}: {type(exc).__name__}: {exc}")
  return tickets

def _exit_timeout_minutes():
  return int(getattr(cfg, 'EXIT_LIMIT_TIMEOUT_MINUTES', 5))

def _exit_reprice_timeout_minutes():
  return int(getattr(cfg, 'EXIT_LIMIT_REPRICE_TIMEOUT_MINUTES', 5))

def _exit_natural_timeout_minutes():
  return int(getattr(cfg, 'EXIT_LIMIT_NATURAL_TIMEOUT_MINUTES', 2))

def _exit_reprice_fraction():
  value = float(getattr(cfg, 'EXIT_LIMIT_REPRICE_FRACTION', 0.5))
  return min(max(value, 0.0), 1.0)

def _close_leg_qtys(structure):
  return [-int(q) for q in structure.get('qtys', [])]

def _combo_exit_limit(algorithm, structure, aggressiveness=0.0):
  symbols = list(structure.get('symbols', []))
  qtys = _close_leg_qtys(structure)
  mid = _signed_mark(algorithm, symbols, qtys)
  natural = _signed_natural(algorithm, symbols, qtys)
  if mid is None:
    return None
  if natural is None:
    natural = mid
  a = min(max(float(aggressiveness), 0.0), 1.0)
  price = float(mid + a * (natural - mid))
  return float(round(price, 2))

def _cancel_close_tickets_intentionally(algorithm, structure, reason):
  ids = list(structure.get('close_ticket_ids', []))
  intentional = set(structure.get('intentional_close_cancel_ids', []))
  for ticket_id in ids:
    ticket = _get_ticket(algorithm, ticket_id)
    if ticket is None or ticket.status not in _ACTIVE_STATUSES:
      continue
    intentional.add(int(ticket_id))
    structure['intentional_close_cancel_ids'] = list(intentional)
    try:
      ticket.cancel(reason)
    except Exception as exc:
      algorithm.error(f"EXIT CANCEL FAILED {structure['ticker']} id={ticket_id}: {type(exc).__name__}: {exc}")
  structure['close_pending'] = False

def _schedule_exit_followup(algorithm, key, cycle, expected_stage, delay_minutes):
  when = pd.Timestamp(algorithm.time) + pd.Timedelta(minutes=int(delay_minutes))
  if when.date() != pd.Timestamp(algorithm.time).date():
    return False
  try:
    date_rule = algorithm.date_rules.on(int(when.year), int(when.month), int(when.day))
    time_rule = algorithm.time_rules.at(int(when.hour), int(when.minute))

    def _callback():
      _process_exit_followup(algorithm, key, int(cycle), str(expected_stage))
    algorithm.schedule.on(date_rule, time_rule, _callback)
    return True
  except Exception as exc:
    algorithm.error(f'EXIT FOLLOWUP SCHEDULE FAILED {key}: {type(exc).__name__}: {exc}')
    return False

def _submit_combo_limit_close(algorithm, key, structure, tag, stage, aggressiveness, timeout_minutes):
  actual = [_holding_qty(algorithm, s) for s in structure['symbols']]
  expected = [int(q) * int(structure['contracts']) for q in structure['qtys']]
  if actual != expected:
    return False
  close_qtys = _close_leg_qtys(structure)
  limit = _combo_exit_limit(algorithm, structure, aggressiveness=aggressiveness)
  if limit is None:
    return False
  legs = [Leg.create(symbol, qty) for symbol, qty in zip(structure['symbols'], close_qtys)]
  tickets = algorithm.combo_limit_order(legs, int(structure['contracts']), float(limit), tag=tag)
  ticket_list = list(tickets) if tickets is not None else []
  ids = _ticket_ids(ticket_list)
  if not ids:
    return False
  structure['close_ticket_ids'] = ids
  structure['close_pending'] = True
  structure['close_tag'] = tag
  structure['close_stage'] = str(stage)
  structure['close_submit_time'] = pd.Timestamp(algorithm.time)
  structure['close_failed'] = False
  scheduled = _schedule_exit_followup(algorithm, key, int(structure['close_cycle']), str(stage), int(timeout_minutes))
  if not scheduled:
    _cancel_close_tickets_intentionally(algorithm, structure, 'Exit follow-up unavailable')
    return False
  return True

def _submit_combo_market_close(algorithm, structure, tag):
  actual = [_holding_qty(algorithm, s) for s in structure['symbols']]
  expected = [int(q) * int(structure['contracts']) for q in structure['qtys']]
  if actual != expected:
    return []
  close_qtys = _close_leg_qtys(structure)
  legs = [Leg.create(symbol, qty) for symbol, qty in zip(structure['symbols'], close_qtys)]
  tickets = algorithm.combo_market_order(legs, int(structure['contracts']), tag=tag)
  return list(tickets) if tickets is not None else []

def _submit_market_fallback(algorithm, structure, tag):
  if not _has_structure_holding(algorithm, structure):
    return []
  actual = [_holding_qty(algorithm, s) for s in structure['symbols']]
  expected = [int(q) * int(structure['contracts']) for q in structure['qtys']]
  ticket_list = []
  if actual == expected and (not structure.get('force_individual_close', False)):
    try:
      ticket_list = _submit_combo_market_close(algorithm, structure, tag)
    except Exception as exc:
      algorithm.error(f"COMBO MARKET EXIT FAILED {structure['ticker']}: {type(exc).__name__}: {exc}")
      structure['force_individual_close'] = True
  if not ticket_list:
    ticket_list = _submit_individual_flatten(algorithm, structure, tag)
    structure['force_individual_close'] = True
  structure['close_ticket_ids'] = _ticket_ids(ticket_list)
  structure['close_pending'] = bool(structure['close_ticket_ids'])
  structure['close_tag'] = tag
  structure['close_stage'] = 'MARKET'
  structure['close_submit_time'] = pd.Timestamp(algorithm.time)
  structure['close_failed'] = False
  if not _has_structure_holding(algorithm, structure):
    structure['closed'] = True
    structure['close_pending'] = False
    if not structure.get('close_confirmed_logged'):
      structure['close_confirmed_logged'] = True
  return ticket_list

def _process_exit_followup(algorithm, key, cycle, expected_stage):
  _prune_registry(algorithm)
  structure = getattr(algorithm, '_vh_open_structures', {}).get(key)
  if structure is None or structure.get('closed'):
    return
  if int(structure.get('close_cycle', -1)) != int(cycle):
    return
  if str(structure.get('close_stage')) != str(expected_stage):
    return
  if not _has_structure_holding(algorithm, structure):
    structure['closed'] = True
    structure['close_pending'] = False
    return
  tag = str(structure.get('close_tag', 'VH_EXIT'))
  _cancel_close_tickets_intentionally(algorithm, structure, f'Advance exit from {expected_stage}')
  actual = [_holding_qty(algorithm, s) for s in structure['symbols']]
  expected = [int(q) * int(structure['contracts']) for q in structure['qtys']]
  if actual != expected:
    structure['force_individual_close'] = True
    _submit_market_fallback(algorithm, structure, tag)
    return
  if expected_stage == 'LIMIT_MID':
    ok = _submit_combo_limit_close(algorithm, key, structure, tag, stage='LIMIT_CROSS', aggressiveness=_exit_reprice_fraction(), timeout_minutes=_exit_reprice_timeout_minutes())
    if ok:
      return
    _submit_market_fallback(algorithm, structure, tag)
    return
  if expected_stage == 'LIMIT_CROSS':
    ok = _submit_combo_limit_close(algorithm, key, structure, tag, stage='LIMIT_NATURAL', aggressiveness=1.0, timeout_minutes=_exit_natural_timeout_minutes())
    if ok:
      return
    _submit_market_fallback(algorithm, structure, tag)
    return
  _submit_market_fallback(algorithm, structure, tag)

def _close(algorithm, key, structure, tag):
  if structure.get('closed'):
    return
  if structure.get('close_pending'):
    if any((_ticket_open(algorithm, i) for i in structure.get('close_ticket_ids', []))):
      return
    structure['close_pending'] = False
  if not _has_structure_holding(algorithm, structure):
    structure['closed'] = True
    return
  structure['close_cycle'] = int(structure.get('close_cycle', 0)) + 1
  structure['close_tag'] = tag
  structure['close_reason'] = tag
  structure['close_failed'] = False
  structure['close_confirmed_logged'] = False
  try:
    actual = [_holding_qty(algorithm, s) for s in structure['symbols']]
    expected = [int(q) * int(structure['contracts']) for q in structure['qtys']]
    can_combo_close = actual == expected and (not structure.get('force_individual_close', False))
    is_stop = '|R=SL|' in tag
    is_force = '|R=FC|' in tag
    if is_stop:
      _submit_market_fallback(algorithm, structure, tag)
      return
    if is_force:
      if can_combo_close:
        ok = _submit_combo_limit_close(algorithm, key, structure, tag, stage='LIMIT_NATURAL', aggressiveness=1.0, timeout_minutes=_exit_natural_timeout_minutes())
        if ok:
          return
      _submit_market_fallback(algorithm, structure, tag)
      return
    if can_combo_close:
      ok = _submit_combo_limit_close(algorithm, key, structure, tag, stage='LIMIT_MID', aggressiveness=0.0, timeout_minutes=_exit_timeout_minutes())
      if ok:
        return
    _submit_market_fallback(algorithm, structure, tag)
  except Exception as exc:
    structure['close_pending'] = False
    structure['close_failed'] = True
    structure['force_individual_close'] = True
    algorithm.error(f"EXIT FAILED {structure['ticker']}: {type(exc).__name__}: {exc}")
    try:
      _submit_market_fallback(algorithm, structure, tag)
    except Exception as fallback_exc:
      algorithm.error(f"EXIT FALLBACK FAILED {structure['ticker']}: {type(fallback_exc).__name__}: {fallback_exc}")

def _repair_failed_entries(algorithm):
  for structure in getattr(algorithm, '_vh_open_structures', {}).values():
    if structure.get('closed') or not structure.get('entry_failed'):
      continue
    if any((_ticket_open(algorithm, i) for i in structure.get('entry_ticket_ids', []))):
      continue
    if not _has_structure_holding(algorithm, structure):
      structure['closed'] = True
      continue
    tickets = _submit_individual_flatten(algorithm, structure, 'ORPHAN_CLEANUP')
    if tickets:
      structure['close_ticket_ids'] = _ticket_ids(tickets)
      structure['close_pending'] = True
      structure['close_tag'] = 'ORPHAN_CLEANUP'
      structure['force_individual_close'] = True

def _assignment_guard(algorithm):
  for ticker, symbol in getattr(algorithm, '_vh_equity_symbols', {}).items():
    try:
      qty = int(algorithm.portfolio[symbol].quantity)
      if qty == 0 or _has_open_order_for_symbol(algorithm, symbol):
        continue
      algorithm.market_order(symbol, -qty, tag='ASSIGNMENT_GUARD')
    except Exception as exc:
      algorithm.error(f'ASSIGNMENT_GUARD FAILED {ticker}: {type(exc).__name__}: {exc}')

def _sync_actual_entry_premium(algorithm, structure):
  if structure.get('actual_entry_synced'):
    return True
  if any((_ticket_open(algorithm, i) for i in structure.get('entry_ticket_ids', []))):
    return False
  if structure.get('entry_failed'):
    return False
  contracts = int(structure.get('contracts', 0) or 0)
  if contracts <= 0:
    return False
  symbols = list(structure.get('symbols', []))
  qtys = [int(q) for q in structure.get('qtys', [])]
  if not symbols or len(symbols) != len(qtys):
    return False
  expected = [q * contracts for q in qtys]
  actual = [_holding_qty(algorithm, s) for s in symbols]
  if actual != expected:
    return False
  signed_package = 0.0
  for symbol, qty_per_contract in zip(symbols, qtys):
    try:
      avg_fill = float(algorithm.portfolio[symbol].average_price or 0.0)
    except Exception:
      return False
    if avg_fill <= 0.0:
      return False
    signed_package += int(qty_per_contract) * avg_fill
  premium_type = str(structure.get('premium_type', '')).upper()
  model_credit = float(structure.get('model_entry_credit', 0.0) or 0.0)
  model_debit = float(structure.get('model_entry_debit', 0.0) or 0.0)
  model_rpc = float(structure.get('model_risk_per_contract', 0.0) or 0.0)
  if premium_type == 'CREDIT':
    actual_credit = max(-signed_package, 0.0)
    if actual_credit <= 0.0:
      return False
    structure['entry_credit'] = actual_credit
    structure['entry_debit'] = 0.0
    delta_risk_per_contract = (model_credit - actual_credit) * cfg.CONTRACT_MULTIPLIER
    structure['actual_entry_price_gap'] = actual_credit - model_credit
  else:
    actual_debit = max(signed_package, 0.0)
    if actual_debit <= 0.0:
      return False
    structure['entry_credit'] = 0.0
    structure['entry_debit'] = actual_debit
    delta_risk_per_contract = (actual_debit - model_debit) * cfg.CONTRACT_MULTIPLIER
    structure['actual_entry_price_gap'] = actual_debit - model_debit
  actual_rpc = max(model_rpc + delta_risk_per_contract, 0.0)
  structure['risk_dollars'] = actual_rpc * contracts
  structure['actual_risk_per_contract'] = actual_rpc
  structure['actual_entry_package_signed'] = float(signed_package)
  structure['actual_entry_synced'] = True
  return True

def _position_snapshot(algorithm, structure):
  now = pd.Timestamp(algorithm.time).normalize()
  remaining = int((structure['expiry'] - now).days)
  mark = _signed_mark(algorithm, structure['symbols'], structure['qtys'])
  if mark is None:
    return None
  contracts = int(structure['contracts'])
  if structure['premium_type'] == 'CREDIT':
    gross = (mark + structure['entry_credit']) * cfg.CONTRACT_MULTIPLIER * contracts
    base = max(structure['entry_credit'] * cfg.CONTRACT_MULTIPLIER * contracts, 1e-09)
  else:
    gross = (mark - structure['entry_debit']) * cfg.CONTRACT_MULTIPLIER * contracts
    base = max(structure['entry_debit'] * cfg.CONTRACT_MULTIPLIER * contracts, 1e-09)
  return {'remaining': remaining, 'mark': float(mark), 'gross': float(gross), 'base': float(base), 'profit_target': float(cfg.PROFIT_TARGET_FRACTION * base), 'stop_loss': float(-cfg.STOP_LOSS_MULTIPLE * base)}

def _update_management_diagnostics(structure, snap):
  gross = float(snap['gross'])
  structure['mgmt_checks'] = int(structure.get('mgmt_checks', 0)) + 1
  structure['last_gross'] = gross
  structure['last_mark'] = float(snap['mark'])
  structure['last_remaining'] = int(snap['remaining'])
  prior_worst = structure.get('worst_gross')
  prior_best = structure.get('best_gross')
  structure['worst_gross'] = gross if prior_worst is None else min(float(prior_worst), gross)
  structure['best_gross'] = gross if prior_best is None else max(float(prior_best), gross)

def _exit_diag_tag(structure, reason, snap=None):
  reason_code = {'PROFIT_TARGET': 'PT', 'STOP_LOSS': 'SL', 'DTE_EXIT': 'DTE', 'FORCE_CLOSE': 'FC'}.get(str(reason), str(reason)[:4].upper())
  d = str(structure.get('signal_date', 'NA'))
  ticker = str(structure.get('ticker', '')).upper()
  checks = int(structure.get('mgmt_checks', 0))
  if snap is None:
    gross = structure.get('last_gross')
    base = None
    remaining = structure.get('last_remaining')
    mark = structure.get('last_mark')
  else:
    gross = snap.get('gross')
    base = snap.get('base')
    remaining = snap.get('remaining')
    mark = snap.get('mark')
  worst = structure.get('worst_gross')
  tag = f'VX|D={d}|T={ticker}|R={reason_code}|G={_fmt_tag_number(gross)}|B={_fmt_tag_number(base)}|W={_fmt_tag_number(worst)}|K={checks}|E={_fmt_tag_number(remaining)}|M={_fmt_tag_number(mark, 2)}'
  return tag[:100]

def _close_with_diagnostics(algorithm, key, structure, reason, snap=None):
  _close(algorithm, key, structure, _exit_diag_tag(structure, reason, snap))

def manage_open_positions(algorithm, force_close=False):
  _prune_registry(algorithm)
  _repair_failed_entries(algorithm)
  for key, structure in list(getattr(algorithm, '_vh_open_structures', {}).items()):
    if structure.get('closed'):
      continue
    if any((_ticket_open(algorithm, i) for i in structure.get('entry_ticket_ids', []))):
      continue
    if structure.get('entry_failed'):
      continue
    actual_synced = _sync_actual_entry_premium(algorithm, structure)
    if not actual_synced and (not force_close):
      continue
    snap = _position_snapshot(algorithm, structure)
    if force_close:
      if snap is not None:
        _update_management_diagnostics(structure, snap)
      _close_with_diagnostics(algorithm, key, structure, 'FORCE_CLOSE', snap)
      continue
    if snap is None:
      continue
    _update_management_diagnostics(structure, snap)
    gross = snap['gross']
    base = snap['base']
    remaining = snap['remaining']
    if gross >= cfg.PROFIT_TARGET_FRACTION * base:
      _close_with_diagnostics(algorithm, key, structure, 'PROFIT_TARGET', snap)
    elif gross <= -cfg.STOP_LOSS_MULTIPLE * base:
      _close_with_diagnostics(algorithm, key, structure, 'STOP_LOSS', snap)
    elif remaining <= cfg.MIN_EXIT_DTE:
      _close_with_diagnostics(algorithm, key, structure, 'DTE_EXIT', snap)
  _assignment_guard(algorithm)
  _prune_registry(algorithm)

def cancel_stale_entry_orders(algorithm, max_age_minutes=None):
  age = int(max_age_minutes or cfg.ENTRY_ORDER_TIMEOUT_MINUTES)
  now = getattr(algorithm, 'utc_time', None)
  if now is None:
    now = algorithm.time
  for structure in getattr(algorithm, '_vh_open_structures', {}).values():
    if structure.get('closed'):
      continue
    canceled_any = False
    for ticket_id in structure.get('entry_ticket_ids', []):
      ticket = _get_ticket(algorithm, ticket_id)
      if ticket is None or ticket.status not in _ACTIVE_STATUSES:
        continue
      try:
        order = algorithm.transactions.get_order_by_id(int(ticket_id))
        created = getattr(order, 'time', None) if order is not None else None
        if created is None:
          algorithm.error(f'ENTRY CANCEL FAILED id={ticket_id}: missing order creation time')
          continue
        minutes = (now - created).total_seconds() / 60.0
        if minutes >= age:
          ticket.cancel('Entry timeout')
          canceled_any = True
      except Exception as exc:
        algorithm.error(f'ENTRY CANCEL FAILED id={ticket_id}: {type(exc).__name__}: {exc}')
    if canceled_any:
      structure['entry_failed'] = True
  _prune_registry(algorithm)

def handle_execution_order_event(algorithm, order_event):
  try:
    status = order_event.status
    if status not in _FAILED_STATUSES:
      return
    order_id = int(order_event.order_id)
    registry = getattr(algorithm, '_vh_open_structures', {})
    for structure in registry.values():
      if order_id in structure.get('entry_ticket_ids', []):
        structure['entry_failed'] = True
        return
      if order_id in structure.get('close_ticket_ids', []):
        intentional = set((int(x) for x in structure.get('intentional_close_cancel_ids', [])))
        if order_id in intentional:
          intentional.discard(order_id)
          structure['intentional_close_cancel_ids'] = list(intentional)
          return
        structure['close_failed'] = True
        structure['close_pending'] = False
        structure['force_individual_close'] = True
        return
  except Exception as exc:
    try:
      algorithm.error(f'ORDER EVENT HANDLER FAILED {type(exc).__name__}: {exc}')
    except Exception:
      pass

