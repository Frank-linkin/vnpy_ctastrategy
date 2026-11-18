"""全部日本蜡烛图形态、RSI 与可配置周期 SMA 策略，周期由输入 K 线决定。"""

from datetime import datetime
from pathlib import Path

import numpy as np
import talib
from talib import abstract

from vnpy.trader.constant import Direction, Interval, Offset, OrderType
from vnpy_ctastrategy import (
    ArrayManager, BarData, BarGenerator, CtaTemplate, OrderData,
    StopOrder, TickData, TradeData,
)
from vnpy_ctastrategy.base import BacktestingMode, EngineType


class JapanPatternWithMaStrategy(CtaTemplate):
    """用 RSI 筛选蜡烛图信号，按持仓时间和均线穿越管理单手仓位。"""

    author = "Codex"
    backtest_from_5m = True
    m: int = 14
    sma_window: int = 10
    rsi_low: float = 30
    rsi_high: float = 70
    rsi_value: float = 0
    sma_value: float = 0
    holding_bars: int = 0
    waiting_cross: bool = False
    parameters = ["m", "rsi_low", "rsi_high", "sma_window"]
    variables = ["rsi_value", "sma_value", "holding_bars", "waiting_cross"]

    def on_init(self) -> None:
        """校验参数并初始化全部形态函数、指标缓存及成交状态。"""
        if not isinstance(self.m, int) or isinstance(self.m, bool) or self.m < 2:
            raise ValueError("m 必须为不小于 2 的整数")
        if (not isinstance(self.sma_window, int) or isinstance(self.sma_window, bool)
                or self.sma_window < 2):
            raise ValueError("sma_window 必须为不小于 2 的整数")
        if not 0 <= self.rsi_low < self.rsi_high <= 100:
            raise ValueError("RSI 阈值必须满足 0 <= rsi_low < rsi_high <= 100")
        if (self.get_engine_type() == EngineType.BACKTESTING
                and self.cta_engine.mode != BacktestingMode.BAR):
            raise ValueError("本策略的下一根开盘成交约定只支持 K 线回测")
        self.patterns = {
            name: abstract.Function(name)
            for name in talib.get_function_groups()["Pattern Recognition"]
        }
        size = max(100, self.m + 2, self.sma_window + 2,
                   max(f.lookback for f in self.patterns.values()) + 2)
        self.am = ArrayManager(size=size)
        self.bg = BarGenerator(self.on_bar)
        self.pending_short_close: float | None = None
        self.pending_short_patterns: list[str] = []
        self.signal_patterns: list[str] = []
        self.entry_time: datetime | None = None
        self.active_orders: set[str] = set()
        self.order_reasons: dict[str, str] = {}
        self.reverse_target = 0
        self.reverse_reason = ""
        self.backtest_log_path: Path | None = None
        if self.get_engine_type() == EngineType.BACKTESTING:
            self.sup_create_backtest_log()
        self.write_log(f"初始化：m={self.m}，RSI阈值={self.rsi_low}/{self.rsi_high}，"
                       f"SMA={self.sma_window}，全部形态数={len(self.patterns)}，预热={size}根")
        warmup_interval = Interval.MINUTE
        if (
            self.get_engine_type() == EngineType.BACKTESTING
            and self.cta_engine.interval in (Interval.MINUTE15, Interval.HOUR)
        ):
            warmup_interval = self.cta_engine.interval
        self.load_bar(10, interval=warmup_interval)

    def sup_create_backtest_log(self) -> None:
        """用本次回测开始的本地时间创建日志文件，同一分钟重复运行时避免覆盖。"""
        directory = Path(r"D:\Project\vnpy_series\data\ctabacktest_log")
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M") + "_japan_pattern_backtest"
        sequence = 1
        while True:
            suffix = "" if sequence == 1 else f"_{sequence}"
            path = directory / f"{stem}{suffix}.log"
            try:
                with path.open("x", encoding="utf-8"):
                    pass
            except FileExistsError:
                sequence += 1
                continue
            self.backtest_log_path = path
            self.cta_engine.output(f"策略回测日志：{path}")
            return

    def write_log(self, msg: str) -> None:
        """保留引擎日志，并将回测日志逐条追加落盘，异常中断前的记录也能保留。"""
        super().write_log(msg)
        path = getattr(self, "backtest_log_path", None)
        if path is not None:
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(self.cta_engine.logs[-1] + "\n")

    def on_start(self) -> None:
        """记录策略启动及输入周期约定。"""
        self.write_log("策略启动；on_bar 接收目标周期K线，Tick输入默认合成1分钟K线")
        self.put_event()

    def on_stop(self) -> None:
        """停止时撤销活动委托并清除待反手目标。"""
        self.reverse_target = 0
        self.cancel_all()
        self.write_log("策略停止，已请求撤销活动委托")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        """将 Tick 合成为分钟K线；小时或日线应由外部聚合后传入 on_bar。"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """计算信号，优先反手，再判断穿越和第十根起的普通出场。"""
        self.am.update_bar(bar)
        if not self.am.inited:
            return
        self.rsi_value = float(self.am.rsi(self.m))
        sma = self.am.sma(self.sma_window, array=True)
        self.sma_value = float(sma[-1])
        if not self.trading:
            # 预热只计算指标，不保留可用于正式交易的历史信号。
            self.pending_short_close = None
            self.pending_short_patterns = []
            self.signal_patterns = []
            return
        if not np.isfinite([self.rsi_value, sma[-1], sma[-2]]).all():
            self.write_log(f"bar_time={bar.datetime} 指标无效，跳过")
            return
        signal = self.sup_get_signal(bar)
        # 有持仓&有实际开仓成交时间&当前k线时间晚于开仓成交时间，就将holding_bars+1
        if self.pos and self.entry_time is not None and bar.datetime > self.entry_time:
            self.holding_bars += 1
        self.write_log(f"bar_time={bar.datetime} close={bar.close_price} RSI={self.rsi_value:.4f} "
                       f"SMA{self.sma_window}={self.sma_value:.4f} pos={self.pos} "
                       f"持仓根数={self.holding_bars} 等待反穿={self.waiting_cross}")
        if self.active_orders:
            self.write_log("仍有活动委托，跳过新的交易决策")
        # 有信号 && （是空仓 或者 是相反信号）
        elif signal and (not self.pos or signal * self.pos < 0):
            self.sup_trade_to(signal, "、".join(self.signal_patterns))
        # 如果没有执行上述开仓或反手操作，而且仍有持仓，就检查均线穿越和普通出场条件。sma[-2] 是上一根K线的SMA值，用来与当前值一起判断是否发生穿越。
        elif self.pos:
            self.sup_manage_position(bar, float(sma[-2]))
        # 向界面发送状态更新事件，使仓位、指标和持仓根数等显示最新值。
        self.put_event()

    def sup_get_signal(self, bar: BarData) -> int:
        """扫描所有形态；空头信号仅由紧接着的一根收盘价确认。"""
        bullish: list[str] = []
        bearish: list[str] = []
        inputs = {"open": self.am.open, "high": self.am.high,
                  "low": self.am.low, "close": self.am.close}
        # 判断属于哪种Candlestick Pattern
        for name, function in self.patterns.items():
            value = function(inputs)[-1]
            if value > 0:
                bullish.append(name)
            elif value < 0:
                bearish.append(name)

        # 对上一根空头信号进行确认
        short_signal = False
        confirmed_patterns = self.pending_short_patterns.copy()
        self.signal_patterns = []
        if self.pending_short_close is not None:
            short_signal = bar.close_price < self.pending_short_close
            self.write_log(f"bar_time={bar.datetime} 空头次根确认：前收={self.pending_short_close} "
                           f"现收={bar.close_price}，结果={short_signal}")
        # 将上一个k线的空头信号清掉
        self.pending_short_close = None
        self.pending_short_patterns = []
        # 本根k线至少出现一个看涨形态，且RSI<= rsi_low, 则设置多头信号
        long_signal = bool(bullish) and self.rsi_value <= self.rsi_low
        if bearish and self.rsi_value >= self.rsi_high:
            self.pending_short_close = bar.close_price
            self.pending_short_patterns = bearish.copy()
            self.write_log(f"bar_time={bar.datetime} 空头形态={bearish}，等待下一根确认")
        if long_signal:
            self.write_log(f"bar_time={bar.datetime} 多头形态={bullish}，RSI={self.rsi_value:.4f}")
        if long_signal and short_signal:
            self.write_log(f"bar_time={bar.datetime} 多空开仓信号冲突，跳过本次入场")
            return 0
        # 多头信号，本根看涨，返回1
        # 空头信号，本根确认看空，返回-1
        # 无信号，返回 0
        self.signal_patterns = bullish.copy() if long_signal else confirmed_patterns if short_signal else []
        return 1 if long_signal else -1 if short_signal else 0

    def sup_manage_position(self, bar: BarData, previous_sma: float) -> None:
        """前十根首次逆向穿越后等待反穿，否则第十根起逐根检查出场。"""
        previous_close = self.am.close[-2]
        cross_up = previous_close <= previous_sma and bar.close_price > self.sma_value
        cross_down = previous_close >= previous_sma and bar.close_price < self.sma_value
        adverse_cross = cross_down if self.pos > 0 else cross_up
        return_cross = cross_up if self.pos > 0 else cross_down
        if self.waiting_cross:
            if return_cross:
                self.sup_trade_to(0, "均线穿越后回归")
            return
        if 1 <= self.holding_bars <= 10 and adverse_cross:
            self.waiting_cross = True
            self.write_log("前10根出现首次逆向穿越，保留持仓至反向穿越；暂停普通出场")
            return
        if self.holding_bars >= 10:
            rsi_exit = (self.rsi_value <= self.rsi_low if self.pos > 0
                        else self.rsi_value >= self.rsi_high)
            ma_exit = (bar.close_price <= self.sma_value if self.pos > 0
                       else bar.close_price >= self.sma_value)
            if rsi_exit or ma_exit:
                reasons = []
                if ma_exit:
                    reasons.append("十日未穿越均线原则出场")
                if rsi_exit:
                    reasons.append("十根K线后RSI条件出")
                self.sup_trade_to(0, "、".join(reasons))

    def sup_trade_to(self, target: int, reason: str) -> None:
        """调整到空仓或单手目标；实盘反手先平仓成交再开仓。"""
        self.write_log(f"交易决策：目标={target}，当前={self.pos}，原因={reason}")
        if self.pos:
            self.reverse_target = target
            self.reverse_reason = reason
            direction = Direction.SHORT if self.pos > 0 else Direction.LONG
            close_reason = "触发新的相反信号" if target else reason
            ids = self.sup_submit_order(direction, Offset.CLOSE, abs(self.pos), close_reason)
            if not ids:
                self.reverse_target = 0
                return
            if self.get_engine_type() == EngineType.LIVE:
                return
            # 回测预先按平仓、开仓顺序排队，两笔均在下一根开盘撮合。
            self.reverse_target = 0
        if target:
            direction = Direction.LONG if target > 0 else Direction.SHORT
            self.sup_submit_order(direction, Offset.OPEN, 1, reason)

    def sup_submit_order(
        self, direction: Direction, offset: Offset, volume: float, reason: str = ""
    ) -> list[str]:
        """回测用跨价限价模拟下一开盘成交，实盘向柜台发送真正市价单。"""
        if not self.trading:
            return []
        if self.get_engine_type() == EngineType.BACKTESTING:
            # 仅在回测使用边界价格，不读取未来K线；标准撮合以次根open成交。
            price = 1e12 if direction == Direction.LONG else self.get_pricetick()
            ids = self.send_order(direction, offset, price, volume)
        else:
            engine = self.cta_engine
            contract = engine.main_engine.get_contract(self.vt_symbol)
            if contract is None:
                self.write_log("市价委托失败：找不到合约")
                return []
            ids = engine.send_server_order(
                self, contract, direction, offset, 0, volume, OrderType.MARKET, False, False
            )
        self.active_orders.update(ids)
        for vt_orderid in ids:
            self.order_reasons[vt_orderid] = reason
        self.write_log(f"发出委托：{direction.value}/{offset.value} 数量={volume} 单号={ids}")
        if not ids:
            self.write_log("委托发送失败，本次交易未执行")
        return ids

    def on_order(self, order: OrderData) -> None:
        """跟踪委托结束状态并记录拒单、撤单和成交进度。"""
        self.write_log(f"委托更新：{order.vt_orderid} {order.status.value} "
                       f"成交={order.traded}/{order.volume}")
        if not order.is_active():
            self.active_orders.discard(order.vt_orderid)
            if order.traded < order.volume:
                self.reverse_target = 0
                self.write_log("委托未完全成交即结束，清除待反手目标")

    def on_trade(self, trade: TradeData) -> None:
        """按实际开仓成交时间重置计数，实盘平仓完成后执行反手开仓。"""
        if trade.offset == Offset.OPEN:
            operation = "开多" if trade.direction == Direction.LONG else "开空"
        else:
            operation = "平空" if trade.direction == Direction.LONG else "平多"
        reason = self.order_reasons.get(trade.vt_orderid, "未找到原委托触发原因")
        self.write_log(f"成交时间={trade.datetime} | 操作={operation} | 触发信号={reason} | "
                       f"价格={trade.price} | 数量={trade.volume} | 当前仓位={self.pos}")
        if trade.offset == Offset.OPEN:
            self.entry_time = trade.datetime
            self.holding_bars = 0
            self.waiting_cross = False
            self.pending_short_close = None
            self.pending_short_patterns = []
        elif not self.pos:
            self.entry_time = None
            self.holding_bars = 0
            self.waiting_cross = False
            if self.reverse_target and self.trading:
                target = self.reverse_target
                self.reverse_target = 0
                self.sup_trade_to(target, self.reverse_reason)
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """记录停止单回报；本策略不主动使用停止单。"""
        self.write_log(f"停止单回报：{stop_order.stop_orderid} {stop_order.status.value}")
