from AlgorithmImports import *
from datetime import date
import qc_config_v2 as cfg
# Load the exit-only executable-natural / individual-flatten overlay.
import execution_v2_hotfix  # noqa: F401
from workflow_v2 import run_workflow
from execution_policy_v2 import manage_open_positions, cancel_stale_entry_orders, handle_execution_order_event


class VolHarvestingThesisV6(QCAlgorithm):
    def initialize(self):
        self.set_start_date(*cfg.BACKTEST_START)
        self.set_end_date(*cfg.BACKTEST_END)
        self.set_cash(cfg.START_CASH)
        self.settings.seed_initial_prices=True

        self._vh_equity_symbols={}; self._vh_option_symbols={}; self._vh_last_slice=None
        self._vh_fallback_option_symbols={}; self._vh_fallback_underlying_symbols={}
        self._vh_open_structures={}; self._vh_iv_history={}; self._vh_last_research={}
        self._vh_entry_day=None; self._vh_entries_today=0; self._vh_last_research_date=None
        self._vh_force_flat_date=date(*cfg.BACKTEST_END)

        for ticker in cfg.TICKERS:
            try:
                eq=self.add_equity(ticker,Resolution.MINUTE); eq.set_data_normalization_mode(DataNormalizationMode.RAW)
                op=self.add_option(ticker,Resolution.MINUTE)
                lo=cfg.FALLBACK_MIN_DTE if ticker=='SPY' and cfg.ENABLE_LIQUID_FALLBACK else cfg.OPTION_MIN_DTE
                hi=max(cfg.OPTION_MAX_DTE,cfg.FALLBACK_MAX_DTE) if ticker=='SPY' and cfg.ENABLE_LIQUID_FALLBACK else cfg.OPTION_MAX_DTE
                if ticker=='SPY' and cfg.ENABLE_LIQUID_FALLBACK:
                    op.set_filter(lambda u,lo=lo,hi=hi:u.include_weeklys().strikes(-cfg.OPTION_STRIKE_RANGE,cfg.OPTION_STRIKE_RANGE).expiration(lo,hi))
                else:
                    op.set_filter(lambda u,lo=lo,hi=hi:u.strikes(-cfg.OPTION_STRIKE_RANGE,cfg.OPTION_STRIKE_RANGE).expiration(lo,hi))
                try: op.price_model=OptionPriceModels.crank_nicolson_fd()
                except Exception: pass
                self._vh_equity_symbols[ticker]=eq.symbol; self._vh_option_symbols[ticker]=op.symbol
                if ticker=='SPY' and cfg.ENABLE_LIQUID_FALLBACK:
                    self._vh_fallback_underlying_symbols['SPY']=eq.symbol
                    self._vh_fallback_option_symbols['SPY']=op.symbol
            except Exception as exc:
                self.debug(f"SUBSCRIPTION_FAILED|{ticker}|{type(exc).__name__}")

        # NDX short-dated fallback uses the NDXP target chain.  It is isolated
        # from the primary universe and is consulted only when no executable
        # primary structure survives the liquidity gate.
        if cfg.ENABLE_LIQUID_FALLBACK and 'NDX' in cfg.FALLBACK_TICKERS:
            try:
                ndx=self.add_index('NDX',Resolution.MINUTE)
                ndxop=self.add_index_option(ndx.symbol,cfg.NDX_OPTION_TARGET,Resolution.MINUTE)
                ndxop.set_filter(lambda u:u.include_weeklys().strikes(-cfg.OPTION_STRIKE_RANGE,cfg.OPTION_STRIKE_RANGE).expiration(cfg.FALLBACK_MIN_DTE,cfg.FALLBACK_MAX_DTE))
                try: ndxop.price_model=OptionPriceModels.crank_nicolson_fd()
                except Exception: pass
                self._vh_equity_symbols['NDX']=ndx.symbol
                self._vh_fallback_underlying_symbols['NDX']=ndx.symbol
                self._vh_fallback_option_symbols['NDX']=ndxop.symbol
            except Exception as exc:
                self.debug(f"SUBSCRIPTION_FAILED|NDX_FALLBACK|{type(exc).__name__}|{exc}")

        anchor=self._vh_equity_symbols.get(cfg.ANCHOR,next(iter(self._vh_equity_symbols.values())))
        self.schedule.on(self.date_rules.every_day(anchor),self.time_rules.after_market_open(anchor,40),self._cancel_stale_entries)
        self.schedule.on(self.date_rules.every_day(anchor),self.time_rules.before_market_close(anchor,45),self._manage)
        self.schedule.on(self.date_rules.every_day(anchor),self.time_rules.before_market_close(anchor,15),self._cancel_and_retry_flatten)
        self.schedule.on(self.date_rules.every_day(anchor),self.time_rules.before_market_close(anchor,5),self._final_flatten_retry)
        self.debug(f"V6_INIT|window={cfg.RUN_WINDOW}|period={cfg.BACKTEST_START}->{cfg.BACKTEST_END}|tickers={len(self._vh_equity_symbols)}|fallback={list(self._vh_fallback_option_symbols.keys())}")

    def on_data(self,data):
        self._vh_last_slice=data
        # V6: MC evaluates management once per trading session, so live/backtest
        # management does the same via the scheduled 45-min-before-close check.
        # Do NOT run the 5-minute manager here; that was the V5 model/live mismatch.
        # Current Slice at 10:00 is required so IV is hydrated.  Run once/day.
        if self.time.hour!=10 or self.time.minute!=0: return
        d=self.time.date()
        if self._vh_last_research_date==d: return
        self._vh_last_research_date=d
        if d < date(*cfg.TRADE_START): return
        try:
            run_workflow(self,data)
        except Exception as exc:
            self.error(f"V6_WORKFLOW_FAILED|{self.time}|{type(exc).__name__}|{exc}")
            raise

    def _cancel_stale_entries(self):
        try: cancel_stale_entry_orders(self)
        except Exception as exc: self.error(f"ENTRY_TIMEOUT_MANAGER_FAILED|{type(exc).__name__}|{exc}")

    def _manage(self):
        try: manage_open_positions(self,self.time.date()>=self._vh_force_flat_date)
        except Exception as exc: self.error(f"MANAGE_FAILED|{type(exc).__name__}|{exc}")

    def _cancel_and_retry_flatten(self):
        try:
            cancel_stale_entry_orders(self)
            if self.time.date()>=self._vh_force_flat_date: manage_open_positions(self,True)
        except Exception as exc: self.error(f"FLATTEN_FAILED|{type(exc).__name__}|{exc}")

    def _final_flatten_retry(self):
        if self.time.date()<self._vh_force_flat_date: return
        try: manage_open_positions(self,True)
        except Exception as exc: self.error(f"FINAL_FLATTEN_FAILED|{type(exc).__name__}|{exc}")

    def on_order_event(self,order_event):
        try: handle_execution_order_event(self,order_event)
        except Exception as exc:
            try:self.error(f"ON_ORDER_EVENT_FAILED|{type(exc).__name__}|{exc}")
            except Exception:pass

    def on_end_of_algorithm(self):
        # No orders here. Print only compact research counts.
        r=getattr(self,'_vh_last_research',{}) or {}
        try:
            self.debug(f"V6_END|window={cfg.RUN_WINDOW}|signals_last={len(r.get('signals',[]))}|strategies_last={len(r.get('strategies',[]))}|hjb_last={len(r.get('hjb',[]))}")
        except Exception: pass


