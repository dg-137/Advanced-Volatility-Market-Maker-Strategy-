
import logging
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase

class AdvancedVolatilityMarketMaker(ScriptStrategyBase):
   

    # Basic strategy parameters
    base_bid_spread = 0.0005
    base_ask_spread = 0.0005
    order_refresh_time = 10
    base_order_amount = 0.02
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    base, quote = trading_pair.split('-')

    # Candles configuration
    candle_exchange = "binance"
    candles_interval = "1m"
    candles_length = 30
    max_records = 1000

    # Volatility parameters
    volatility_scalar = 100
    max_volatility_spread = 0.002  # 20 bps max spread
    min_volatility_spread = 0.0001  # 1 bp min spread

    # VWAP Bollinger Bands parameters
    vwap_period = 20
    bb_std_dev = 2.0
    trend_strength_threshold = 0.5

    # Inventory management parameters
    target_inventory_ratio = 0.5
    max_inventory_deviation = 0.3
    inventory_adjustment_factor = 0.8

    # Momentum parameters
    rsi_period = 14
    momentum_threshold = 0.2
    max_momentum_shift = 0.001  # 10 bps max momentum shift

    # Risk management parameters
    max_order_size_ratio = 0.1  # Max 10% of balance per order
    atr_period = 14
    position_size_scalar = 0.5

    # Performance tracking
    total_trades = 0
    total_pnl = 0
    last_inventory_check = 0

    # Initialize candles
    candles = CandlesFactory.get_candle(CandlesConfig(
        connector=candle_exchange,
        trading_pair=trading_pair,
        interval=candles_interval,
        max_records=max_records
    ))

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.candles.start()

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            self.update_strategy_parameters()
            proposal: List[OrderCandidate] = self.create_proposal()
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def get_enhanced_candles(self):
        """Calculate all technical indicators"""
        candles_df = self.candles.candles_df.copy()

        if len(candles_df) < max(self.candles_length, self.vwap_period, self.rsi_period):
            return candles_df

        # Calculate VWAP
        candles_df['vwap'] = (candles_df['close'] * candles_df['volume']).cumsum() / candles_df['volume'].cumsum()

        # Calculate rolling VWAP for Bollinger Bands
        candles_df['rolling_vwap'] = candles_df['close'].rolling(window=self.vwap_period).mean()
        candles_df['vwap_std'] = candles_df['close'].rolling(window=self.vwap_period).std()
        candles_df['vwap_bb_upper'] = candles_df['rolling_vwap'] + (self.bb_std_dev * candles_df['vwap_std'])
        candles_df['vwap_bb_lower'] = candles_df['rolling_vwap'] - (self.bb_std_dev * candles_df['vwap_std'])

        # Calculate NATR for volatility
        candles_df.ta.natr(length=self.candles_length, scalar=1, append=True)

        # Calculate RSI for momentum
        candles_df.ta.rsi(length=self.rsi_period, append=True)

        # Calculate ATR for risk management
        candles_df.ta.atr(length=self.atr_period, append=True)

        # Trend analysis
        candles_df['trend_signal'] = self.calculate_trend_signal(candles_df)

        return candles_df

    def calculate_trend_signal(self, df):
        """Calculate trend signal based on VWAP position and momentum"""
        if len(df) < self.vwap_period:
            return 0

        latest_close = df['close'].iloc[-1]
        vwap_upper = df['vwap_bb_upper'].iloc[-1]
        vwap_lower = df['vwap_bb_lower'].iloc[-1]
        vwap_mid = df['rolling_vwap'].iloc[-1]

        # Position relative to VWAP bands
        if latest_close > vwap_upper:
            return 1  # Strong uptrend
        elif latest_close < vwap_lower:
            return -1  # Strong downtrend
        elif latest_close > vwap_mid:
            return 0.5  # Mild uptrend
        else:
            return -0.5  # Mild downtrend

    def calculate_volatility_spread(self, candles_df):
        """Calculate dynamic spread based on volatility"""
        if f"NATR_{self.candles_length}" not in candles_df.columns:
            return self.base_bid_spread, self.base_ask_spread

        volatility = candles_df[f"NATR_{self.candles_length}"].iloc[-1]

        # Scale volatility to spread
        vol_spread = volatility * self.volatility_scalar

        # Apply bounds
        vol_spread = max(self.min_volatility_spread, 
                        min(self.max_volatility_spread, vol_spread))

        # Asymmetric spreads based on inventory
        inventory_ratio = self.get_inventory_ratio()

        # If we have too much base asset, widen ask spread
        if inventory_ratio > self.target_inventory_ratio:
            bid_spread = vol_spread * 0.8
            ask_spread = vol_spread * 1.2
        # If we have too much quote asset, widen bid spread
        elif inventory_ratio < self.target_inventory_ratio:
            bid_spread = vol_spread * 1.2
            ask_spread = vol_spread * 0.8
        else:
            bid_spread = ask_spread = vol_spread

        return bid_spread, ask_spread

    def get_inventory_ratio(self):
        """Calculate current inventory ratio"""
        try:
            base_balance = self.connectors[self.exchange].get_balance(self.base)
            quote_balance = self.connectors[self.exchange].get_balance(self.quote)

            if base_balance == 0 and quote_balance == 0:
                return self.target_inventory_ratio

            current_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )

            base_value = base_balance * current_price
            total_value = base_value + quote_balance

            if total_value == 0:
                return self.target_inventory_ratio

            return float(base_value / total_value)

        except Exception:
            return self.target_inventory_ratio

    def calculate_momentum_adjustment(self, candles_df):
        """Calculate price shift based on momentum"""
        if f"RSI_{self.rsi_period}" not in candles_df.columns:
            return 0

        rsi = candles_df[f"RSI_{self.rsi_period}"].iloc[-1]
        trend_signal = candles_df['trend_signal'].iloc[-1]

        # Combine RSI with trend signal
        rsi_normalized = (rsi - 50) / 50  # Normalize RSI to [-1, 1]

        # Apply momentum threshold
        if abs(rsi_normalized) < self.momentum_threshold:
            momentum_factor = 0
        else:
            momentum_factor = rsi_normalized * trend_signal

        # Scale to price adjustment
        momentum_shift = momentum_factor * self.max_momentum_shift

        return momentum_shift

    def calculate_risk_adjusted_order_size(self, candles_df):
        """Calculate order size based on volatility and account balance"""
        try:
            # Get current balances
            base_balance = self.connectors[self.exchange].get_balance(self.base)
            quote_balance = self.connectors[self.exchange].get_balance(self.quote)

            current_price = self.connectors[self.exchange].get_price_by_type(
                self.trading_pair, self.price_source
            )

            # Calculate total balance in quote terms
            total_balance = quote_balance + (base_balance * current_price)

            if total_balance == 0:
                return self.base_order_amount

            # Base order size as percentage of total balance
            base_size = total_balance * self.max_order_size_ratio

            # Adjust for volatility
            if f"ATR_{self.atr_period}" in candles_df.columns:
                atr = candles_df[f"ATR_{self.atr_period}"].iloc[-1]
                current_price_float = float(current_price)

                # Lower volatility = larger position size
                volatility_factor = 1 / (1 + (atr / current_price_float))
                adjusted_size = base_size * volatility_factor * self.position_size_scalar
            else:
                adjusted_size = base_size * self.position_size_scalar

            return max(0.001, min(adjusted_size, base_size))

        except Exception:
            return self.base_order_amount

    def update_strategy_parameters(self):
        """Update all strategy parameters based on current market conditions"""
        candles_df = self.get_enhanced_candles()

        if len(candles_df) < self.candles_length:
            return

        # Update spreads
        self.current_bid_spread, self.current_ask_spread = self.calculate_volatility_spread(candles_df)

        # Update momentum adjustment
        self.momentum_adjustment = self.calculate_momentum_adjustment(candles_df)

        # Update order size
        self.current_order_amount = self.calculate_risk_adjusted_order_size(candles_df)

        # Update inventory metrics
        self.current_inventory_ratio = self.get_inventory_ratio()

    def create_proposal(self) -> List[OrderCandidate]:
        """Create optimized order proposals"""
        current_price = self.connectors[self.exchange].get_price_by_type(
            self.trading_pair, self.price_source
        )

        # Apply momentum adjustment to reference price
        adjusted_price = current_price * Decimal(str(1 + self.momentum_adjustment))

        # Calculate order prices
        buy_price = adjusted_price * Decimal(str(1 - self.current_bid_spread))
        sell_price = adjusted_price * Decimal(str(1 + self.current_ask_spread))

        # Ensure orders don't cross the spread
        best_bid = self.connectors[self.exchange].get_price(self.trading_pair, False)
        best_ask = self.connectors[self.exchange].get_price(self.trading_pair, True)

        buy_price = min(buy_price, best_bid)
        sell_price = max(sell_price, best_ask)

        # Create orders
        orders = []

        # Buy order
        buy_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY,
            amount=Decimal(str(self.current_order_amount)),
            price=buy_price
        )
        orders.append(buy_order)

        # Sell order
        sell_order = OrderCandidate(
            trading_pair=self.trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=Decimal(str(self.current_order_amount)),
            price=sell_price
        )
        orders.append(sell_order)

        return orders

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust order sizes to available budget"""
        return self.connectors[self.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=True
        )

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place the orders"""
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place individual order"""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events"""
        self.total_trades += 1
        trade_pnl = float(event.trade_fee.amount) * -1  # Approximate PnL
        self.total_pnl += trade_pnl

        msg = (
            f"{event.trade_type.name} {round(event.amount, 4)} {event.trading_pair} "
            f"at {round(event.price, 4)} | Total trades: {self.total_trades}"
        )
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        """Enhanced status display"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        # Basic information
        lines.extend(["", " ADVANCED VOLATILITY MARKET MAKER "])
        lines.extend(["", f"Trading Pair: {self.trading_pair} | Exchange: {self.exchange}"])

        # Account balances
        balance_df = self.get_balance_df()
        lines.extend(["", "Balances:"])
        lines.extend(["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Active orders
        try:
            orders_df = self.active_orders_df()
            lines.extend(["", " Active Orders:"])
            lines.extend(["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", " No active orders"])

        # Strategy metrics
        lines.extend(["", " Strategy Metrics:"])
        lines.extend([f"    Current Spreads: Bid {self.current_bid_spread*10000:.2f}bps | Ask {self.current_ask_spread*10000:.2f}bps"])
        lines.extend([f"    Momentum Adjustment: {self.momentum_adjustment*10000:.2f}bps"])
        lines.extend([f"    Order Amount: {self.current_order_amount:.4f} {self.base}"])
        lines.extend([f"    Inventory Ratio: {self.current_inventory_ratio:.2%} (Target: {self.target_inventory_ratio:.2%})"])
        lines.extend([f"    Total Trades: {self.total_trades} | Est. PnL: {self.total_pnl:.4f}"])

        # Market data
        candles_df = self.get_enhanced_candles()
        if len(candles_df) > 0:
            lines.extend(["", "Market Analysis:"])
            if f"NATR_{self.candles_length}" in candles_df.columns:
                natr = candles_df[f"NATR_{self.candles_length}"].iloc[-1]
                lines.extend([f"    Volatility (NATR): {natr:.4f}"])
            if f"RSI_{self.rsi_period}" in candles_df.columns:
                rsi = candles_df[f"RSI_{self.rsi_period}"].iloc[-1]
                lines.extend([f"    RSI: {rsi:.2f}"])
            if 'trend_signal' in candles_df.columns:
                trend = candles_df['trend_signal'].iloc[-1]
                trend_desc = "Strong Bull" if trend == 1 else "Strong Bear" if trend == -1 else "Mild Bull" if trend == 0.5 else "Mild Bear" if trend == -0.5 else "Neutral"
                lines.extend([f"    Trend Signal: {trend_desc} ({trend:.2f})"])

        return "\n".join(lines)

# Create the main function to return the strategy
def main():
    return AdvancedVolatilityMarketMaker
