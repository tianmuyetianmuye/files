import pandas as pd
import mplfinance as mpf
from binance.client import Client
from binance.enums import *
import time
import sys
import datetime

# --- Configuration ---
# IMPORTANT: Replace with your actual API Key and Secret from Binance.
# Ensure these keys have Futures Trading permissions enabled.
# Make sure to replace the ENTIRE string including "YOUR_BINANCE_API_KEY"
# Example: API_KEY = "your_actual_api_key_here"
API_KEY = "FABdy60xZxDdAEqY4gj7dtMX83N9B6l0vdGej48aDVhaAHrnExgzbrFF8I0x0LeF"  # <<< IMPORTANT: Replace with your actual API Key
API_SECRET = "U2NBrXdkj58sVZdFeDV6ishey6TjozRBoVX0FhiQ51w8eaoA6pNuGQnDFAoM4JZP"  # <<< IMPORTANT: Replace with your actual API Secret

# --- User Inputs ---
def get_user_inputs():
    """Prompts the user for trading parameters."""
    print("\n--- Futures Trading Bot Setup ---")
    symbol = input("Enter Futures Trading Pair (e.g., BTCUSDT): ").upper()
    
    valid_intervals = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH
    }
    interval_str = input(f"Enter Timeframe ({', '.join(valid_intervals.keys())}): ").lower()
    interval = valid_intervals.get(interval_str)
    while interval is None:
        print("Invalid timeframe. Please choose from the list.")
        interval_str = input(f"Enter Timeframe ({', '.join(valid_intervals.keys())}): ").lower()
        interval = valid_intervals.get(interval_str)

    try:
        usd_size = float(input("Enter Position Size in USD (e.g., 100): "))
        if usd_size <= 0:
            raise ValueError
    except ValueError:
        print("Invalid USD size. Please enter a positive number.")
        sys.exit()

    bias = input("Choose Bias (long/short): ").lower()
    while bias not in ['long', 'short']:
        print("Invalid bias. Please enter 'long' or 'short'.")
        bias = input("Choose Bias (long/short): ").lower()

    return symbol, interval, usd_size, bias

# Global variable to track current position
current_position = None # Can be None, 'LONG', or 'SHORT'

# --- Binance API Functions ---
def get_binance_klines(client, symbol, interval, lookback_milliseconds):
    """
    Fetches historical kline (candlestick) data from Binance Futures.
    Args:
        client (Client): The Binance API client instance.
        symbol (str): The futures trading pair (e.g., 'BTCUSDT').
        interval (str): The kline interval (e.g., Client.KLINE_INTERVAL_15MINUTE).
        lookback_milliseconds (int): Milliseconds to look back from server time.
    Returns:
        list: A list of kline data.
    """
    try:
        # print("Attempting to get server time...") # Commented for less verbose output
        server_time_ms = client.get_server_time()['serverTime']
        # print(f"Server time obtained: {server_time_ms}") # Commented for less verbose output
        start_time_ms = server_time_ms - lookback_milliseconds
        # print(f"Fetching klines for {symbol} from {datetime.datetime.fromtimestamp(start_time_ms / 1000)} UTC...") # Commented for less verbose output
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time_ms)
        # print(f"Received {len(klines)} klines.") # Commented for less verbose output
        return klines
    except Exception as e:
        print(f"Error fetching klines from Binance Futures: {e}")
        print("Action: Please check your API keys, network connection, or Binance API status.")
        return []

def process_klines_to_dataframe(klines):
    """
    Processes raw kline data into a Pandas DataFrame suitable for mplfinance.
    Args:
        klines (list): Raw kline data from Binance.
    Returns:
        pd.DataFrame: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    if not klines:
        # print("No klines provided to process_klines_to_dataframe.") # Commented for less verbose output
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = 'Date'
    # print(f"DataFrame processed. Head:\n{df.head()}") # Commented for less verbose output
    return df

def get_current_price(client, symbol):
    """
    Fetches the current market price for a given symbol.
    Args:
        client (Client): The Binance API client instance.
        symbol (str): The trading pair symbol.
    Returns:
        float: The current price, or None if an error occurs.
    """
    try:
        # print(f"Attempting to get current price for {symbol}...") # Commented for less verbose output
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        # print(f"Current price for {symbol}: {price}") # Commented for less verbose output
        return price
    except Exception as e:
        print(f"Error getting current price for {symbol}: {e}")
        print("Action: Ensure the symbol is correct and your API can access ticker data.")
        return None

def calculate_quantity(client, symbol, usd_size, price):
    """
    Calculates the quantity of coins to trade based on USD size and current price.
    Considers symbol's lot size (stepSize) for precision.
    Args:
        client (Client): The Binance API client instance.
        symbol (str): The trading pair symbol.
        usd_size (float): Desired position size in USD.
        price (float): Current price of the asset.
    Returns:
        float: The calculated quantity, rounded to the correct step size, or None on error.
    """
    if price is None or price == 0:
        print("Cannot calculate quantity: Price is invalid or zero.")
        return None

    try:
        # print(f"Attempting to get exchange info for {symbol}...") # Commented for less verbose output
        info = client.futures_exchange_info()
        symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)

        if not symbol_info:
            print(f"Error: Could not find exchange info for symbol: {symbol}")
            print("Action: Double-check the symbol spelling.")
            return None

        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            print(f"Error: Could not find LOT_SIZE filter for {symbol}")
            print("Action: This is unusual; try a different symbol or check Binance API docs.")
            return None

        step_size = float(lot_size_filter['stepSize'])
        # print(f"Step size for {symbol}: {step_size}") # Commented for less verbose output
        raw_quantity = usd_size / price
        rounded_quantity = round(raw_quantity / step_size) * step_size
        
        if rounded_quantity <= 0:
            print(f"Calculated quantity is zero or negative ({rounded_quantity}). Check USD size or price.")
            return None

        print(f"Calculated quantity: {rounded_quantity} for {usd_size} USD at {price:.2f}")
        return rounded_quantity
    except Exception as e:
        print(f"Error calculating quantity for {symbol}: {e}")
        print("Action: Check Binance API documentation for symbol filters or try a different symbol.")
        return None

def place_market_order(client, symbol, side, quantity, reduce_only=False):
    """
    Places a market order on Binance Futures.
    Args:
        client (Client): The Binance API client instance.
        symbol (str): The trading pair symbol.
        side (str): Order side (e.g., SIDE_BUY, SIDE_SELL).
        quantity (float): The quantity of the asset to trade.
        reduce_only (bool): Set to True for closing positions (reduces existing position).
    Returns:
        dict: The order response, or None on failure.
    """
    global current_position
    print(f"Attempting to place market order: Symbol={symbol}, Side={side}, Quantity={quantity}, ReduceOnly={reduce_only}")
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=reduce_only,
            newOrderRespType='FULL'
        )
        print(f"Order placed successfully: {order}")
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        print("Action: Check your Binance Futures API key permissions (ensure 'Enable Futures' is checked), "
              "account balance, and current position. Also check if you're hitting rate limits.")
        return None

def plot_candlestick_chart(df, symbol, interval):
    """
    Plots a candlestick chart using mplfinance, including multiple Moving Averages and the latest price.
    Args:
        df (pd.DataFrame): DataFrame containing OHLCV data.
        symbol (str): The trading pair symbol.
        interval (str): The kline interval.
    """
    if df.empty:
        print("No data to plot for chart. Skipping chart generation.")
        return

    # Calculate Moving Averages
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['100_MA'] = df['Close'].rolling(window=100).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()

    # Get the latest closing price
    latest_price = df['Close'].iloc[-1]
    chart_title = (
        f"{symbol} Candlestick Chart (Futures - {interval})\n"
        f"Latest Price: {latest_price:.2f} | "
        f"MA(20): {df['20_MA'].iloc[-1]:.2f} | "
        f"MA(50): {df['50_MA'].iloc[-1]:.2f} | "
        f"MA(100): {df['100_MA'].iloc[-1]:.2f} | "
        f"MA(200): {df['200_MA'].iloc[-1]:.2f}"
    )

    print(f"Generating initial chart for {symbol} ({interval}) with MAs...")

    # Define the style for the chart
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        volume='in',
        ohlc='i'
    )
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        gridcolor='gray',
        gridstyle=':',
        y_on_right=True,
        rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'text.color': 'white'},
        facecolor='#1a1a1a' # Dark background
    )

    # Create addplots for all MAs
    apds = [
        mpf.make_addplot(df['20_MA'], color='blue', panel=0, width=0.7, linestyle='-', label='20 MA'),
        mpf.make_addplot(df['50_MA'], color='orange', panel=0, width=0.7, linestyle='-', label='50 MA'),
        mpf.make_addplot(df['100_MA'], color='purple', panel=0, width=0.7, linestyle='-', label='100 MA'),
        mpf.make_addplot(df['200_MA'], color='brown', panel=0, width=0.7, linestyle='-', label='200 MA')
    ]

    # Plot the chart
    mpf.plot(
        df,
        type='candle',
        style=s,
        title=chart_title, # Use the updated title with latest price and MAs
        ylabel='Price',
        ylabel_lower='Volume',
        volume=True,
        figratio=(16,9), # Aspect ratio
        tight_layout=True,
        addplot=apds, # Add all moving average plots
        # legend_loc='best' # mplfinance automatically adds legend if labels are provided
    )
    print("Initial chart generated successfully.")


# --- Main Trading Logic ---
def run_trading_bot():
    global current_position

    symbol, interval, usd_size, bias = get_user_inputs()
    
    try:
        client = Client(API_KEY, API_SECRET)
        print("Binance Client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Binance Client: {e}")
        print("Action: Double-check your API_KEY and API_SECRET. They might be incorrect or have issues.")
        sys.exit() # Exit if client cannot be initialized

    # Determine lookback based on interval to ensure enough data for 200 MA
    # We need at least 200 candles for the 200 MA, plus some buffer.
    interval_ms_map = {
        Client.KLINE_INTERVAL_1MINUTE: 60 * 1000,
        Client.KLINE_INTERVAL_3MINUTE: 3 * 60 * 1000,
        Client.KLINE_INTERVAL_5MINUTE: 5 * 60 * 1000,
        Client.KLINE_INTERVAL_15MINUTE: 15 * 60 * 1000,
        Client.KLINE_INTERVAL_30MINUTE: 30 * 60 * 1000,
        Client.KLINE_INTERVAL_1HOUR: 60 * 60 * 1000,
        Client.KLINE_INTERVAL_2HOUR: 2 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_4HOUR: 4 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_6HOUR: 6 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_8HOUR: 8 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_12HOUR: 12 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_1DAY: 24 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_3DAY: 3 * 24 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_1WEEK: 7 * 24 * 60 * 60 * 1000,
        Client.KLINE_INTERVAL_1MONTH: 30 * 24 * 60 * 60 * 1000 # Approximate
    }
    # Fetch enough data for 200 MA + some buffer (e.g., 250 candles worth)
    lookback_candles = 250 # Increased lookback to ensure enough data for 200 MA
    lookback_milliseconds = lookback_candles * interval_ms_map.get(interval, 15 * 60 * 1000) # Default to 15m if not found

    print(f"\nStarting bot for {symbol} on {interval} timeframe with {usd_size} USD size ({bias} bias).")
    print("Monitoring for MA conditions...")

    # --- Initial Plot ---
    print("Fetching initial data for chart display...")
    raw_klines_initial = get_binance_klines(client, symbol, interval, lookback_milliseconds)
    df_initial = process_klines_to_dataframe(raw_klines_initial)
    # Ensure enough data for 200 MA and for iloc[-2] (last completed bar)
    if not df_initial.empty and len(df_initial) >= 200: # Need at least 200 candles for 200 MA
        plot_candlestick_chart(df_initial, symbol, interval)
    else:
        print("Not enough initial data to plot chart. Proceeding with bot logic only.")
        if df_initial.empty:
            print("Initial data fetch failed or returned empty.")
        elif len(df_initial) < 200:
            print(f"Only {len(df_initial)} candles fetched initially, need at least 200 for 200 MA and trading logic.")


    while True:
        try:
            raw_klines = get_binance_klines(client, symbol, interval, lookback_milliseconds)
            df = process_klines_to_dataframe(raw_klines)

            if df.empty:
                print("No data received. Retrying in 1 second...")
                time.sleep(1)
                continue

            # Ensure there are enough data points for all MAs and for iloc[-1] (current bar's open)
            # The longest MA is 200, so we need at least 200 candles for it to be calculated.
            # We also need at least 1 candle for the current bar's open.
            if len(df) < 200:
                print(f"Not enough data for all MAs ({len(df)} candles). Waiting for more data...")
                time.sleep(1)
                continue
            
            # We need at least 1 candle for the current bar's open price (iloc[-1])
            if len(df) < 1: # This check is technically redundant if len(df) < 200 passes, but good for clarity
                print(f"Not enough data for open price logic ({len(df)} candles). Waiting for more data...")
                time.sleep(1)
                continue

            # Calculate Moving Averages based on all available data (including the current, incomplete bar's close)
            # For the purpose of comparing the *open* of the current bar, we need the MA values
            # that were valid *at the close of the previous bar*.
            df['20_MA'] = df['Close'].rolling(window=20).mean()
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['100_MA'] = df['Close'].rolling(window=100).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()


            # Current real-time price for monitoring (last incomplete bar's current close)
            current_realtime_price = df['Close'].iloc[-1]
            current_realtime_20ma = df['20_MA'].iloc[-1]


            # Data for trading logic (using the OPEN of the current bar and MAs from the PREVIOUS bar)
            current_bar_open = df['Open'].iloc[-1] # The open price of the bar that just started
            
            # MAs at the close of the PREVIOUS completed bar (iloc[-2])
            # This is crucial because the decision is made at the open of the *new* bar,
            # based on the MAs that were fully formed from the *previous* bar's close.
            ma_20_prev_close = df['20_MA'].iloc[-2]
            ma_50_prev_close = df['50_MA'].iloc[-2]
            ma_100_prev_close = df['100_MA'].iloc[-2]
            ma_200_prev_close = df['200_MA'].iloc[-2]


            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Current Price: {current_realtime_price:.2f}, Current 20 MA: {current_realtime_20ma:.2f}, "
                  f"Position: {current_position if current_position else 'FLAT'}")

            # --- Trading Logic ---
            if bias == 'long':
                # Entry condition for long bias: Open of current bar is above all MAs (from previous bar's close)
                long_entry_condition = (
                    current_bar_open > ma_20_prev_close and
                    current_bar_open > ma_50_prev_close and
                    current_bar_open > ma_100_prev_close and
                    current_bar_open > ma_200_prev_close and
                    current_position is None
                )

                if long_entry_condition:
                    print(f"LONG signal detected: Open of current bar ({current_bar_open:.2f}) is above all MAs.")
                    # Execute at current market price
                    current_market_price_for_order = get_current_price(client, symbol)
                    if current_market_price_for_order:
                        quantity = calculate_quantity(client, symbol, usd_size, current_market_price_for_order)
                        if quantity:
                            order_response = place_market_order(client, symbol, SIDE_BUY, quantity, reduce_only=False)
                            if order_response:
                                current_position = 'LONG'
                                print(f"Entered LONG position. Quantity: {quantity}")
                # Exit condition for long bias: Open of current bar is below all MAs (from previous bar's close)
                elif (current_bar_open < ma_20_prev_close and
                      current_bar_open < ma_50_prev_close and
                      current_bar_open < ma_100_prev_close and
                      current_bar_open < ma_200_prev_close) and \
                     current_position == 'LONG':
                    print(f"EXIT LONG signal detected: Open of current bar ({current_bar_open:.2f}) is below all MAs.")
                    # Execute at current market price
                    current_market_price_for_order = get_current_price(client, symbol)
                    if current_market_price_for_order:
                        quantity = calculate_quantity(client, symbol, usd_size, current_market_price_for_order)
                        if quantity:
                            order_response = place_market_order(client, symbol, SIDE_SELL, quantity, reduce_only=True)
                            if order_response:
                                current_position = None
                                print(f"Exited LONG position. Quantity: {quantity}")

            elif bias == 'short':
                # Entry condition for short bias: Open of current bar is below all MAs (from previous bar's close)
                short_entry_condition = (
                    current_bar_open < ma_20_prev_close and
                    current_bar_open < ma_50_prev_close and
                    current_bar_open < ma_100_prev_close and
                    current_bar_open < ma_200_prev_close and
                    current_position is None
                )

                if short_entry_condition:
                    print(f"SHORT signal detected: Open of current bar ({current_bar_open:.2f}) is below all MAs.")
                    # Execute at current market price
                    current_market_price_for_order = get_current_price(client, symbol)
                    if current_market_price_for_order:
                        quantity = calculate_quantity(client, symbol, usd_size, current_market_price_for_order)
                        if quantity:
                            order_response = place_market_order(client, symbol, SIDE_SELL, quantity, reduce_only=False)
                            if order_response:
                                current_position = 'SHORT'
                                print(f"Entered SHORT position. Quantity: {quantity}")
                # Exit condition for short bias: Open of current bar is above all MAs (from previous bar's close)
                elif (current_bar_open > ma_20_prev_close and
                      current_bar_open > ma_50_prev_close and
                      current_bar_open > ma_100_prev_close and
                      current_bar_open > ma_200_prev_close) and \
                     current_position == 'SHORT':
                    print(f"EXIT SHORT signal detected: Open of current bar ({current_bar_open:.2f}) is above all MAs.")
                    # Execute at current market price
                    current_market_price_for_order = get_current_price(client, symbol)
                    if current_market_price_for_order:
                        quantity = calculate_quantity(client, symbol, usd_size, current_market_price_for_order)
                        if quantity:
                            order_response = place_market_order(client, symbol, SIDE_BUY, quantity, reduce_only=True)
                            if order_response:
                                current_position = None
                                print(f"Exited SHORT position. Quantity: {quantity}")

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            print("Action: Review the error message and the preceding print statements for clues.")

        time.sleep(1) # Wait for 1 second before the next iteration

if __name__ == "__main__":
    run_trading_bot()
