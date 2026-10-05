import pandas as pd
import qc_config_v2 as cfg
import execution_v2 as ex
from AlgorithmImports import OrderStatus


def _match_row(sized, structure):
    if sized is None or sized.empty:
        return None
    try:
        expiry=pd.Timestamp(structure.get('expiry')).normalize()
        x=sized[
            (sized['ticker'].astype(str)==str(structure.get('ticker')))&
            (pd.to_datetime(sized['expiry']).dt.normalize()==expiry)&
            (sized['structure'].astype(str)==str(structure.get('structure')))
        ]
        return None if x.empty else x.iloc[0]
    except Exception:
        return None


def submit_hjb_combo_orders(algorithm, sized, execute_orders=True):
    before=set(getattr(algorithm,'_vh_open_structures',{}).keys())
    tickets=ex.submit_hjb_combo_orders(algorithm,sized,execute_orders=execute_orders)
    registry=getattr(algorithm,'_vh_open_structures',{})
    for key in set(registry.keys())-before:
        structure=registry[key]
        row=_match_row(sized,structure)
        value=cfg.MIN_EXIT_DTE if row is None else row.get('min_exit_dte',cfg.MIN_EXIT_DTE)
        try: structure['min_exit_dte']=int(value)
        except Exception: structure['min_exit_dte']=int(cfg.MIN_EXIT_DTE)
        structure['risk_class']=str(structure.get('risk_class','CAPPED')).upper()
    return tickets


def _ensure_leg_state(algorithm, structure):
    if structure.get('leg_states_initialized'):
        return True
    if not structure.get('actual_entry_synced'):
        return False
    contracts=int(structure.get('contracts',0) or 0)
    if contracts<=0:
        return False
    states=[]
    mult=float(cfg.CONTRACT_MULTIPLIER)
    for i,(symbol,qpc) in enumerate(zip(structure.get('symbols',[]),structure.get('qtys',[]))):
        try:
            px=float(algorithm.portfolio[symbol].average_price or 0.0)
        except Exception:
            return False
        if px<=0.0:
            return False
        qty=int(qpc)*contracts
        basis=abs(qty)*px*mult
        states.append({
            'index':i,'symbol':symbol,'qty_per_contract':int(qpc),'entry_qty':qty,
            'entry_price':px,'entry_basis':basis,'realized_pnl':0.0,
            'trail_floor_pct':None,'next_target_pct':float(cfg.LEG_TRAIL_START_PCT),
            'exit_order_ids':[],'closed':False,
        })
    if not states:
        return False
    structure['leg_states']=states
    structure['leg_states_initialized']=True
    structure['gross_entry_basis']=float(sum(s['entry_basis'] for s in states))
    structure['largest_leg_entry_basis']=float(max(s['entry_basis'] for s in states))
    structure['strategy_risk_basis']=max(
        float(structure.get('risk_dollars',0.0) or 0.0),
        structure['gross_entry_basis'],
    )
    structure['strategy_trail_floor_dollars']=None
    structure['strategy_next_target_dollars']=max(
        float(cfg.STRATEGY_TRAIL_START_PCT)*structure['strategy_risk_basis'],
        float(cfg.LEG_TRAIL_START_PCT)*structure['largest_leg_entry_basis'],
    )
    structure['strategy_stop_triggered']=False
    return True


def _leg_close_price(algorithm, symbol, holding_qty):
    sec=algorithm.securities[symbol]
    bid=float(sec.bid_price or 0.0); ask=float(sec.ask_price or 0.0)
    if bid<=0.0 or ask<=0.0 or ask<bid:
        return None
    return bid if holding_qty>0 else ask


def _snapshot(algorithm, structure):
    if not _ensure_leg_state(algorithm,structure):
        return None
    mult=float(cfg.CONTRACT_MULTIPLIER)
    realized=0.0; unrealized=0.0; legs=[]
    for state in structure['leg_states']:
        symbol=state['symbol']; entry=float(state['entry_price'])
        realized_leg=float(state.get('realized_pnl',0.0) or 0.0)
        realized+=realized_leg
        try: hqty=int(algorithm.portfolio[symbol].quantity)
        except Exception: hqty=0
        leg_unreal=0.0
        if hqty!=0:
            px=_leg_close_price(algorithm,symbol,hqty)
            if px is None:
                return None
            leg_unreal=hqty*(float(px)-entry)*mult
            unrealized+=leg_unreal
        basis=max(float(state.get('entry_basis',0.0) or 0.0),1e-9)
        leg_total=realized_leg+leg_unreal
        legs.append({
            'index':int(state['index']),'symbol':symbol,'holding_qty':hqty,
            'pnl':float(leg_total),'return_pct':float(leg_total/basis),
            'basis':basis,'state':state,
        })
    gross=float(realized+unrealized)
    base=max(float(structure.get('strategy_risk_basis',structure.get('gross_entry_basis',0.0)) or 0.0),1e-9)
    remaining=int((pd.Timestamp(structure['expiry']).normalize()-pd.Timestamp(algorithm.time).normalize()).days)
    return {'remaining':remaining,'mark':0.0,'gross':gross,'base':base,
            'gross_return_pct':gross/base,'legs':legs}


def _risk_reducing_leg_exit(structure, leg):
    # Buying back a short leg always removes exposure.  Selling a protective
    # long wing can create naked risk, so that condition escalates to full exit.
    return int(leg['state'].get('qty_per_contract',0))<0


def _submit_leg_exit(algorithm, structure, leg, reason):
    state=leg['state']
    if state.get('closed') or state.get('exit_order_ids'):
        return False
    qty=int(leg.get('holding_qty',0) or 0)
    if qty==0:
        state['closed']=True
        return False
    try:
        ticket=algorithm.market_order(state['symbol'],-qty,tag=f"VL|T={structure.get('ticker')}|I={state['index']}|R={reason}"[:100])
        if ticket is None:
            return False
        oid=int(ticket.order_id)
        state['exit_order_ids']=[oid]
        structure.setdefault('leg_exit_order_map',{})[oid]=int(state['index'])
        return True
    except Exception as exc:
        algorithm.error(f"LEG_EXIT_FAILED|{structure.get('ticker')}|{type(exc).__name__}|{exc}")
        return False


def _advance_leg_trail(algorithm, structure, leg):
    state=leg['state']; r=float(leg['return_pct'])
    floor=state.get('trail_floor_pct')
    nxt=float(state.get('next_target_pct',cfg.LEG_TRAIL_START_PCT))
    just_raised=False
    if floor is None:
        if r>=nxt:
            state['trail_floor_pct']=nxt
            state['next_target_pct']=nxt+float(cfg.TRAIL_STEP_PCT)
            just_raised=True
            algorithm.debug(f"LEG_TRAIL_ARM|{structure.get('ticker')}|i={state['index']}|floor={nxt:.2f}|next={state['next_target_pct']:.2f}")
    else:
        while r>=float(state.get('next_target_pct',nxt)):
            state['trail_floor_pct']=float(state['next_target_pct'])
            state['next_target_pct']=float(state['next_target_pct'])+float(cfg.TRAIL_STEP_PCT)
            just_raised=True
        if (not just_raised) and r<=float(state['trail_floor_pct']):
            return True
    return False


def _advance_strategy_trail(algorithm, structure, snap):
    gross=float(snap['gross']); base=float(snap['base'])
    floor=structure.get('strategy_trail_floor_dollars')
    nxt=float(structure.get('strategy_next_target_dollars',0.0) or 0.0)
    just_raised=False
    if floor is None:
        if nxt>0.0 and gross>=nxt:
            structure['strategy_trail_floor_dollars']=nxt
            structure['strategy_next_target_dollars']=nxt+float(cfg.TRAIL_STEP_PCT)*base
            just_raised=True
            algorithm.debug(f"STRAT_TRAIL_ARM|{structure.get('ticker')}|floor={nxt:.2f}|next={structure['strategy_next_target_dollars']:.2f}")
    else:
        while gross>=float(structure.get('strategy_next_target_dollars',nxt)):
            structure['strategy_trail_floor_dollars']=float(structure['strategy_next_target_dollars'])
            structure['strategy_next_target_dollars']=float(structure['strategy_next_target_dollars'])+float(cfg.TRAIL_STEP_PCT)*base
            just_raised=True
        if (not just_raised) and gross<=float(structure['strategy_trail_floor_dollars']):
            return True
    return False


def _hard_stop_if_needed(algorithm,key,structure,snap):
    loss_pct=-float(snap['gross_return_pct'])
    hard=float(cfg.STRATEGY_HARD_STOP_PCT)
    trigger=float(cfg.STRATEGY_STOP_TRIGGER_PCT)
    if loss_pct>=hard:
        try:
            ex._cancel_close_tickets_intentionally(algorithm,structure,'Hard stop')
        except Exception:
            pass
        structure['force_individual_close']=True
        ex._submit_market_fallback(algorithm,structure,ex._exit_diag_tag(structure,'STOP_LOSS',snap))
        return True
    if loss_pct>=trigger:
        if not structure.get('close_pending'):
            structure['strategy_stop_triggered']=True
            ex._close_with_diagnostics(algorithm,key,structure,'STRATEGY_STOP_TRIGGER',snap)
        return True
    return False


def manage_open_positions(algorithm, force_close=False):
    ex._prune_registry(algorithm)
    ex._repair_failed_entries(algorithm)
    for key,structure in list(getattr(algorithm,'_vh_open_structures',{}).items()):
        if structure.get('closed'): continue
        # V6 cadence alignment: research/MC first observes a position after one
        # session has elapsed.  Avoid a same-day 15:15 stop on a 10:00 entry.
        if (not force_close) and str(getattr(cfg,'MANAGEMENT_FREQUENCY','DAILY_CLOSE')).upper()=='DAILY_CLOSE':
            try:
                signal_day=pd.to_datetime(str(structure.get('signal_date','')),format='%y%m%d',errors='coerce')
                if pd.notna(signal_day) and pd.Timestamp(algorithm.time).normalize()<=signal_day.normalize():
                    continue
            except Exception:
                pass
        if any(ex._ticket_open(algorithm,i) for i in structure.get('entry_ticket_ids',[])): continue
        if structure.get('entry_failed'): continue
        actual_synced=ex._sync_actual_entry_premium(algorithm,structure)
        if not actual_synced and not force_close: continue
        snap=_snapshot(algorithm,structure)
        if force_close:
            if snap is not None: ex._update_management_diagnostics(structure,snap)
            ex._close_with_diagnostics(algorithm,key,structure,'FORCE_CLOSE',snap)
            continue
        if snap is None: continue
        ex._update_management_diagnostics(structure,snap)

        # A -8% strategy drawdown triggers the normal limit-exit ladder; -10%
        # forces the remaining holdings out.  Percentages use gross entry premium,
        # not tiny net fly credit/debit.
        if _hard_stop_if_needed(algorithm,key,structure,snap):
            continue
        if structure.get('close_pending'):
            continue

        if _advance_strategy_trail(algorithm,structure,snap):
            ex._close_with_diagnostics(algorithm,key,structure,'TRAIL_PROFIT',snap)
            continue

        # ASYM_V5 carry positions are managed as packages only.  The C3 ablation
        # showed that the 15% single-leg stop/trail chopped off the put-side carry
        # before the full package risk budget had time to work.  Primary capped
        # flies retain the original leg-level management because this flag is False.
        if not bool(structure.get('package_only_management',False)):
            leg_action=False
            for leg in snap['legs']:
                if int(leg['holding_qty'])==0:
                    continue
                if float(leg['return_pct'])<=-float(cfg.LEG_STOP_LOSS_PCT):
                    if _risk_reducing_leg_exit(structure,leg):
                        leg_action=_submit_leg_exit(algorithm,structure,leg,'LEG_STOP')
                    else:
                        ex._close_with_diagnostics(algorithm,key,structure,'LEG_STOP_ESCALATE',snap)
                        leg_action=True
                    if leg_action: break
                if _advance_leg_trail(algorithm,structure,leg):
                    if _risk_reducing_leg_exit(structure,leg):
                        leg_action=_submit_leg_exit(algorithm,structure,leg,'LEG_TRAIL')
                    else:
                        ex._close_with_diagnostics(algorithm,key,structure,'LEG_TRAIL_ESCALATE',snap)
                        leg_action=True
                    if leg_action: break
            if leg_action:
                continue

        exit_dte=int(structure.get('min_exit_dte',cfg.MIN_EXIT_DTE))
        if int(snap['remaining'])<=exit_dte:
            ex._close_with_diagnostics(algorithm,key,structure,'DTE_EXIT',snap)
    ex._assignment_guard(algorithm)
    ex._prune_registry(algorithm)


def handle_execution_order_event(algorithm,order_event):
    ex.handle_execution_order_event(algorithm,order_event)
    oid=int(order_event.order_id)
    for structure in getattr(algorithm,'_vh_open_structures',{}).values():
        idx=(structure.get('leg_exit_order_map',{}) or {}).get(oid)
        if idx is None:
            continue
        states=structure.get('leg_states',[])
        if int(idx)>=len(states):
            return
        state=states[int(idx)]
        status=getattr(order_event,'status',None)
        if status in (OrderStatus.CANCELED,OrderStatus.INVALID):
            state['exit_order_ids']=[]
            return
        fill_qty=float(getattr(order_event,'fill_quantity',0.0) or 0.0)
        fill_px=float(getattr(order_event,'fill_price',0.0) or 0.0)
        if fill_qty!=0.0 and fill_px>0.0:
            pnl=-fill_qty*(fill_px-float(state['entry_price']))*float(cfg.CONTRACT_MULTIPLIER)
            state['realized_pnl']=float(state.get('realized_pnl',0.0) or 0.0)+float(pnl)
        try:
            if int(algorithm.portfolio[state['symbol']].quantity)==0:
                state['closed']=True
                state['exit_order_ids']=[]
        except Exception:
            pass
        return


cancel_stale_entry_orders=ex.cancel_stale_entry_orders
