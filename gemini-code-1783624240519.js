// backtester_2.js - True Historical Forward/Backtester (Web Service Edition)
const ccxt = require('ccxt');
const { Impit } = require('impit');
const http = require('http'); // 👈 NEW: Required for the dummy server

// Your static weights from Supabase
const settings = {
    ema_fast_period: 5, ema_slow_period: 13,
    macro_ema_fast: 27, macro_ema_slow: 63,
    weight_macd: 3.0, weight_rsi: 1.5, weight_ema: 2.0, 
    weight_pattern: 2.5, weight_history: 1.0, macro_weight: 1.5,
    penalty_3_candles: 2.0, penalty_4_candles: 3.0, penalty_5_candles: 10.0,
    rvol_threshold: 1.5, volatility_threshold: 0.14,
    base_confidence: 50.1, 
    high_volatility_confidence: 50.1 
};

// --- NEW: Crypto.com UpDown Boundary Mechanics ---
const riskSettings = {
    takeProfitPerc: 0.005, // 0.5% Target (Ceiling/Floor Knockout)
    stopLossPerc: 0.005,   // 0.5% Stop (Ceiling/Floor Knockout)
    slippagePerc: 0.0005   // 0.05% assumed entry slippage 
};

// Math Helpers
const calculateEMAArray = (data, period) => {
    const k = 2 / (period + 1);
    let emaArray = [data[0]];
    for (let i = 1; i < data.length; i++) {
        emaArray.push((data[i] * k) + (emaArray[i - 1] * (1 - k)));
    }
    return emaArray;
};

const calculateRSI = (closes) => {
    let gains = [], losses = [];
    for (let i = 1; i < closes.length; i++) {
        let diff = closes[i] - closes[i - 1];
        gains.push(diff > 0 ? diff : 0);
        losses.push(diff < 0 ? Math.abs(diff) : 0);
    }
    let avgGain = gains.slice(0, 14).reduce((a, b) => a + b, 0) / 14;
    let avgLoss = losses.slice(0, 14).reduce((a, b) => a + b, 0) / 14;
    let rsi = avgLoss === 0 ? 100 : 100 - (100 / (1 + (avgGain / avgLoss)));
    for (let i = 14; i < gains.length; i++) {
        avgGain = ((avgGain * 13) + gains[i]) / 14;
        avgLoss = ((avgLoss * 13) + losses[i]) / 14;
        rsi = avgLoss === 0 ? 100 : (avgGain === 0 ? 0 : 100 - (100 / (1 + (avgGain / avgLoss))));
    }
    return rsi;
};

// Core Engine Simulator
function simulatePrediction(candles) {
    if (candles.length < 50) return { pred: "SKIP", conf: 0 };

    const opens = candles.map(c => parseFloat(c[1]));
    const highs = candles.map(c => parseFloat(c[2]));
    const lows = candles.map(c => parseFloat(c[3]));
    const closes = candles.map(c => parseFloat(c[4]));
    const volumes = candles.map(c => parseFloat(c[5]));

    const currentClose = closes[closes.length - 1];
    const prevOpen = opens[opens.length - 2];
    const prevClose = closes[closes.length - 2];
    const prevHigh = highs[highs.length - 2];
    const prevLow = lows[lows.length - 2];

    const rsi = calculateRSI(closes);
    
    const emaFast = calculateEMAArray(closes, settings.ema_fast_period).pop();
    const emaSlow = calculateEMAArray(closes, settings.ema_slow_period).pop();
    const macroEmaFast = calculateEMAArray(closes, settings.macro_ema_fast).pop();
    const macroEmaSlow = calculateEMAArray(closes, settings.macro_ema_slow).pop();
    
    const ema12 = calculateEMAArray(closes, 12);
    const ema26 = calculateEMAArray(closes, 26);
    const macdLine = ema12.map((v, i) => v - ema26[i]);
    const signalLine = calculateEMAArray(macdLine, 9);
    const currentHist = macdLine[macdLine.length - 1] - signalLine[signalLine.length - 1];
    const prevHist = macdLine[macdLine.length - 2] - signalLine[signalLine.length - 2];

    const volSMA20 = volumes.slice(-20).reduce((a,b)=>a+b,0) / 20;
    const rvol = volumes[volumes.length - 1] / volSMA20;

    const upperWick = prevHigh - Math.max(prevOpen, prevClose);
    const lowerWick = Math.min(prevOpen, prevClose) - prevLow;
    const bodySize = Math.max(Math.abs(prevClose - prevOpen), 0.0001);

    let colorFlips = 0;
    for (let i = closes.length - 1; i >= closes.length - 4; i--) {
        const currentColor = closes[i] >= opens[i] ? 'green' : 'red';
        const prevColor = closes[i-1] >= opens[i-1] ? 'green' : 'red';
        if (currentColor !== prevColor) colorFlips++;
    }
    const isWhipsaw = colorFlips >= 3;

    let trSum = 0;
    for (let i = closes.length - 14; i < closes.length; i++) {
        const highLow = highs[i] - lows[i];
        const highClose = Math.abs(highs[i] - closes[i-1]);
        const lowClose = Math.abs(lows[i] - closes[i-1]);
        trSum += Math.max(highLow, highClose, lowClose);
    }
    const atrPercentage = ((trSum / 14) / currentClose) * 100;

    let pred = "SKIP";
    
    if (rsi < 43) {
        pred = "UP"; 
    } else if (rsi > 57) {
        pred = "DOWN"; 
    } else {
        return { pred: "SKIP", conf: 0 };
    }

    let isVetoed = false;
    if (pred === "UP" && currentHist < -0.1 && currentHist < prevHist) isVetoed = true;
    if (pred === "DOWN" && currentHist > 0.1 && currentHist > prevHist) isVetoed = true;
    if (atrPercentage < 0.05) isVetoed = true; 

    if (isVetoed) return { pred: "SKIP", conf: 0 };

    let conf = 45.0; 
    
    if (pred === "UP" && macroEmaFast > macroEmaSlow) conf += 6.0;
    if (pred === "DOWN" && macroEmaFast < macroEmaSlow) conf += 6.0;

    if (pred === "UP" && emaFast > emaSlow) conf += 4.0;
    if (pred === "DOWN" && emaFast < emaSlow) conf += 4.0;

    if (pred === "UP" && lowerWick > bodySize) conf += 3.0;
    if (pred === "DOWN" && upperWick > bodySize) conf += 3.0;

    if (rvol > settings.rvol_threshold) conf += 3.0; 
    
    if (atrPercentage > settings.volatility_threshold) {
        conf -= 6.0;
    } else {
        conf += 2.0;
    }
    
    if (isWhipsaw) conf -= 6.0; 

    if (conf < settings.base_confidence) return { pred: "SKIP", conf };

    return { pred, conf };
}

async function runBacktest() {
    const impersonator = new Impit({ 
        browser: 'chrome',
        proxyUrl: 'http://zirrujpi-ch-city_wettingen-954674:8e2wprq017db@p.webshare.io:80' 
    });

    try {
        console.log("🌐 Verifying connection routing...");
        const res = await impersonator.fetch('https://api.ipify.org?format=json');
        const data = await res.json();
        console.log("✅ Current Bot IP:", data.ip);
    } catch (err) {
        console.error("❌ Failed to verify proxy IP. Check credentials:", err.message);
        return; 
    }

    console.log("📥 Fetching historical 5m candles for ETH/USD...");
    
    const exchange = new ccxt.cryptocom({
        fetchImplementation: impersonator.fetch.bind(impersonator)
    });
    
    let allCandles = [];
    let since = exchange.milliseconds() - (3000 * 5 * 60 * 1000); 

    try {
        while (allCandles.length < 3000) {
            const batch = await exchange.fetchOHLCV('ETH/USD', '5m', since, 1000);
            if (batch.length === 0) break;
            allCandles = allCandles.concat(batch);
            since = batch[batch.length - 1][0] + 1;
            console.log(`...fetched ${allCandles.length} candles`);
            await new Promise(r => setTimeout(r, 1000));
        }
    } catch (e) {
        console.log("Fetch Error:", e.message);
    }

    const splitIndex = Math.floor(allCandles.length * 0.7);
    const inSample = allCandles.slice(0, splitIndex);
    const outOfSample = allCandles.slice(splitIndex);

    const testPhase = (dataArray, phaseName) => {
        let wins = 0, losses = 0, skips = 0;
        let position = null; 
        
        for (let i = 50; i < dataArray.length - 1; i++) {
            const currentCandle = dataArray[i];
            const high = currentCandle[2];
            const low = currentCandle[3];
            const close = currentCandle[4];

            if (position) {
                let tradeClosed = false;

                if (position.type === 'UP') {
                    if (low <= position.sl) {
                        losses++;
                        tradeClosed = true;
                    } else if (high >= position.tp) {
                        wins++;
                        tradeClosed = true;
                    }
                } 
                else if (position.type === 'DOWN') {
                    if (high >= position.sl) {
                        losses++;
                        tradeClosed = true;
                    } else if (low <= position.tp) {
                        wins++;
                        tradeClosed = true;
                    }
                }

                if (tradeClosed) {
                    position = null; 
                }
                continue; 
            }

            const historicalSlice = dataArray.slice(i - 50, i);
            const { pred } = simulatePrediction(historicalSlice);
            
            if (pred === "SKIP") {
                skips++;
            } else if (pred === "UP") {
                const entryPrice = close * (1 + riskSettings.slippagePerc);
                position = {
                    type: 'UP',
                    entry: entryPrice,
                    tp: entryPrice * (1 + riskSettings.takeProfitPerc),
                    sl: entryPrice * (1 - riskSettings.stopLossPerc)
                };
            } else if (pred === "DOWN") {
                const entryPrice = close * (1 - riskSettings.slippagePerc);
                position = {
                    type: 'DOWN',
                    entry: entryPrice,
                    tp: entryPrice * (1 - riskSettings.takeProfitPerc),
                    sl: entryPrice * (1 + riskSettings.stopLossPerc)
                };
            }
        }

        const totalTrades = wins + losses;
        const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
        
        console.log(`\n📊 --- ${phaseName} RESULTS ---`);
        console.log(`Total Trades Executed: ${totalTrades}`);
        console.log(`Wins: ${wins} | Losses: ${losses} | Skipped: ${skips}`);
        console.log(`Strict Win Rate: ${winRate}%`);
    };

    testPhase(inSample, "IN-SAMPLE (Sandbox Phase)");
    testPhase(outOfSample, "OUT-OF-SAMPLE (Lie Detector Phase)");
}

// 👈 NEW: Dummy HTTP Server to satisfy Render's Web Service Health Check
const PORT = process.env.PORT || 3000;
http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Crypto.com Backtester is alive! Check your Render Logs to see the results.\n');
}).listen(PORT, () => {
    console.log(`🟢 Dummy web server bound to port ${PORT}. Render Health Checks will pass!`);
    
    // Kick off the backtest once the server is successfully listening
    runBacktest().catch(console.error);
});