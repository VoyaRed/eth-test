import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import tensorflow as tf
import xgboost as xgb
import requests
import itertools
import time
from datetime import datetime

# --- NEW IMPORTS FOR RENDER WORKAROUND ---
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from supabase import create_client, Client
import os

# Initialize Supabase
SUPABASE_URL = "https://bpizkikscieyhzajrrrg.supabase.co"
SUPABASE_KEY = "sb_publishable_UwkvMmq91Y5jS1PEJ9IE_w_vN5-JWUP"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_prediction_to_supabase(action, probability, price):
    # Insert initial prediction
    data = supabase.table("bot_predictions").insert({
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "price": float(price),
        "prediction": action,
        "probability": float(probability),
        "status": "pending" 
    }).execute()
    return data.data[0]['id'] # Returns the ID to update later

def resolve_prediction(pred_id, entry_price):
    # Fetch price after 5 minutes
    df_new = fetch_latest_data(SYMBOL, TIMEFRAME, 2)
    exit_price = df_new['close'].iloc[-1]
    
    # Logic: If predicted LONG and price > entry_price, it's a WIN
    is_win = exit_price > entry_price
    
    supabase.table("bot_predictions").update({
        "status": "resolved",
        "result": "win" if is_win else "loss",
        "exit_price": float(exit_price)
    }).eq("id", pred_id).execute()
    print(f"✅ Prediction {pred_id} resolved! Result: {'WIN' if is_win else 'LOSS'}")

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1524667437872582676/OQcFu_e-ZLl-NYo3qxyz81tB-vVXy-7FapTPEol205G8ZBIkby5RcpzWzFW7OUzUqvtq"
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
THRESHOLD = 0.60
LIMIT = 150  # Fetch enough candles to satisfy the 108-period EMA + rolling features

# Proxy list from your original notebook
PROXY_LIST = [
    'http://zirrujpi-ch-532845:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532846:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532847:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532848:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532849:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532850:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532851:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532852:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532853:8e2wprq017db@p.webshare.io:80',
    'http://zirrujpi-ch-532854:8e2wprq017db@p.webshare.io:80',
]
proxy_pool = itertools.cycle(PROXY_LIST)

# ==========================================
# 1. LOAD MODELS
# ==========================================
print("🧠 Loading models...")
lstm_model = tf.keras.models.load_model('lstm_binary_extractor.keras')
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_binary_model.json')

# ==========================================
# 2. DUMMY WEB SERVER (RENDER WORKAROUND)
# ==========================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Respond with HTTP 200 OK to keep Render happy
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Crypto Bot is up and running!")

    # Suppress server logging to keep your console clean
    def log_message(self, format, *args):
        pass

def keep_alive_server():
    # Render assigns a port dynamically via the PORT env variable
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"🌐 Started dummy web server on port {port} to satisfy Render.")
    server.serve_forever()

# ==========================================
# 3. FETCH LIVE DATA WITH PROXY ROTATION
# ==========================================
def fetch_latest_data(symbol, timeframe, limit):
    print(f"📡 Fetching latest {limit} candles for {symbol}...")
    
    for _ in range(len(PROXY_LIST)):
        current_proxy = next(proxy_pool)
        print(f"🔄 Attempting connection via proxy: {current_proxy.split('@')[-1]}")
        
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'proxies': {
                    'http': current_proxy,
                    'https': current_proxy
                }
            })
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                raise ValueError("No data returned from exchange.")
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print("✅ Data fetched successfully!")
            return df
            
        except Exception as e:
            print(f"⚠️ Proxy failed ({type(e).__name__}): {e}")
            time.sleep(1.5)
            
    raise Exception("❌ All proxies failed. Could not fetch live market data.")

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    print("📊 Calculating indicators...")
    df['atr'] = df.ta.atr(length=14)
    df['rsi'] = df.ta.rsi(length=14)
    
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    df['rsi_slope'] = df['rsi'].diff(3)
    df['macd_hist_diff'] = df['macd_hist'].diff(1)
    
    df['dist_ema_fast'] = df['close'] - df.ta.ema(length=9)
    df['dist_ema_slow'] = df['close'] - df.ta.ema(length=21)
    df['dist_ema_1H'] = df['close'] - df.ta.ema(length=108)
    
    bbands = df.ta.bbands(length=20, std=2)
    bbw_col = [col for col in bbands.columns if col.startswith('BBB')][0]
    df['bb_width'] = bbands[bbw_col] 
    
    df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['roc3'] = df.ta.roc(length=3)
    df['is_doji'] = (df['body_size'] < (df['atr'] * 0.1)).astype(int)
    
    is_green = (df['close'] > df['open']).astype(int)
    color_flip = (is_green != is_green.shift(1)).astype(int)
    df['flips_in_last_4'] = color_flip.rolling(window=4).sum()
    
    direction = np.sign(df['close'] - df['open'])
    df['consecutive_trend'] = direction.groupby((direction != direction.shift()).cumsum()).cumsum()

    df['ret_close'] = df['close'].pct_change() * 100
    df['ret_high'] = ((df['high'] - df['close'].shift()) / df['close'].shift()) * 100
    df['ret_low'] = ((df['low'] - df['close'].shift()) / df['close'].shift()) * 100
    df['vol_pct'] = df['volume'].pct_change()

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ==========================================
# 5. PREDICTION PIPELINE
# ==========================================
def get_prediction(df):
    seq_length = 5
    tabular_columns = [
        'rsi', 'rsi_slope', 'macd', 'macd_hist', 'macd_hist_diff', 
        'dist_ema_fast', 'dist_ema_slow', 'dist_ema_1H', 'bb_width', 'atr', 'rvol', 
        'roc3', 'body_size', 'upper_wick', 'lower_wick', 'is_doji', 
        'flips_in_last_4', 'consecutive_trend'
    ]
    
    last_idx = len(df) - 1
    
    X_seq_live = df[['ret_close', 'ret_high', 'ret_low', 'vol_pct', 'atr']].values[last_idx-seq_length+1:last_idx+1].reshape(1, seq_length, 5)
    X_tab_live = df[tabular_columns].values[last_idx].reshape(1, len(tabular_columns))
    
    lstm_feat_live = lstm_model.predict(X_seq_live, verbose=0)
    X_combined_live = np.hstack((lstm_feat_live, X_tab_live))
    
    dmatrix_live = xgb.DMatrix(X_combined_live)
    probability = xgb_model.predict(dmatrix_live)[0]
    
    action = "LONG" if probability >= THRESHOLD else "SKIP"
    return action, probability

# ==========================================
# 6. DISCORD WEBHOOK
# ==========================================
def send_discord_webhook(action, probability, current_price, atr_value, atr_ma):
    if atr_value > atr_ma:
        choppiness_status = "Less Choppy (Trend favorable)"
        color = 0x00FF00 if action == "LONG" else 0xFFFF00
    else:
        choppiness_status = "More Choppy (Range bound)"
        color = 0xFFA500
        
    embed = {
        "title": f"🤖 5m Bot Prediction Update | {SYMBOL}",
        "description": f"New candle closing in 30 seconds. Here is the latest model evaluation.",
        "color": color,
        "fields": [
            {"name": "🎯 Signal", "value": f"**{action}**", "inline": True},
            {"name": "📊 Confidence", "value": f"{(probability * 100):.2f}%", "inline": True},
            {"name": "💰 Current Price", "value": f"${current_price:.4f}", "inline": True},
            {"name": "🌊 Market State", "value": choppiness_status, "inline": False}
        ],
        "footer": {
            "text": f"Threshold set at {(THRESHOLD * 100):.1f}% | Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        }
    }
    
    requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})
    print("✅ Webhook executed.")

# ==========================================
# 7. CONTINUOUS EXECUTION LOOP
# ==========================================
def run_bot():
    print("🚀 Bot initialized. Entering monitoring phase...")
    while True:
        current_timestamp = int(datetime.utcnow().timestamp())
        
        # Calculate next 5-minute close and subtract 30 seconds
        next_candle_close = ((current_timestamp // 300) + 1) * 300
        target_timestamp = next_candle_close - 30
        
        sleep_duration = target_timestamp - current_timestamp
        
        if sleep_duration <= 0:
            sleep_duration += 300
            target_timestamp += 300

        target_time_str = datetime.utcfromtimestamp(target_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏳ Sleeping for {sleep_duration} seconds... Waking up at {target_time_str} UTC")
        
        time.sleep(sleep_duration)
        print("⏰ Waking up! 30 seconds to candle close. Executing pipeline...")
        
        try:
            df_live = fetch_latest_data(SYMBOL, TIMEFRAME, LIMIT)
            df_live = engineer_features(df_live)
            
            action, prob = get_prediction(df_live)

            current_price = df_live['close'].iloc[-1]
            current_atr = df_live['atr'].iloc[-1]
            atr_ma = df_live['atr'].rolling(14).mean().iloc[-1]
            
            send_discord_webhook(action, prob, current_price, current_atr, atr_ma)

            # 1. Log the prediction
            pred_id = log_prediction_to_supabase(action, prob, current_price)
            send_discord_webhook(action, prob, current_price, ...)
            
            # 2. Wait 5 minutes to resolve it
            print("⏳ Waiting 5 minutes to resolve prediction...")
            time.sleep(300) 
            
            # 3. Update Supabase
            resolve_prediction(pred_id, current_price)
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            print("🔄 Loop continuing... Will try again next epoch.")

if __name__ == "__main__":
    # 1. Start the dummy web server in a background thread
    server_thread = threading.Thread(target=keep_alive_server, daemon=True)
    server_thread.start()
    
    # 2. Run the main trading loop
    run_bot()
