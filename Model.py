#pip install yfinance
#pip install ta
import ta.momentum
import ta.trend
import ta.volatility
import yfinance as yf
import ta as ta
import sharpe as sharpe
#fetching 5 years of daily data from stock = SPY
data = yf.download("SPY", start="2018-01-01",end="2023-01-01",interval="1d")
#We calculate indicators with ta
'''
1. Trend Indicator: EMA
2. Momentum Indicator: RSI
3. Volatility Indicator: Bollinger Bands
4. Confirmation Indicator: MACD
5. Volume Indicator: OBV
'''
future = 14
#Using EMA14 instead of EMA50
data["EMA14"] = ta.trend.EMAIndicator(data["Adj Close"],window=future).ema_indicator()
#default chosen window=14, can be changed
data["RSI"] = ta.momentum.RSIIndicator(data["Adj Close"],window=future).rsi()
#default chosen window=20, can be changed
bollinger = ta.volatility.BollingerBands(data["Adj Close"],window=future)
data["BB_High"] = bollinger.bollinger_hband()
data["BB_Low"] = bollinger.bollinger_lband()
macd = ta.trend.MACD(data["Adj Close"])
data["MACD"] = macd.macd()
data["MACD_Signal"] = macd.macd_signal()
data["OBV"] = ta.volume.OnBalanceVolumeIndicator(data["Adj Close"], data["Volume"]).on_balance_volume()

data.dropna(inplace=True)
#print(data[["EMA50", "RSI", "BB_High", "BB_Low", "MACD", "MACD_Signal", "OBV"]].head())
#creating labels for model

data["Future_Close"]=data["Adj Close"].shift(future)
#creating thresholds for model to learn
data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], future,True).average_true_range()
data = data.dropna(subset=['ATR'])  # Drop rows where ATR is NaN
data = data[data['ATR'] > 0] 

buy_threshold = data["ATR"] *1.5
sell_threshold = data["ATR"] *0.8
data["Label"] = 0  # Default hold
#data['Sharpe_ratio'] = sharpe.sharpe_ratio(data)

data.loc[((data["Future_Close"] - data["Adj Close"])/data["Adj Close"]) > buy_threshold/data["Adj Close"], "Label"] = 1  # Buy
data.loc[((data["Future_Close"] - data["Adj Close"])/data["Adj Close"]) < -sell_threshold/data["Adj Close"], "Label"] = -1  # Sell
buy_sell_hold_counts = data['Label'].value_counts()

# Print the counts of Buy, Sell, and Hold
print("Buy, Sell, Hold Counts:")
print(buy_sell_hold_counts)


#split data in train/test 
from sklearn.model_selection import train_test_split

X,y = data[["EMA14","RSI","BB_High","BB_Low","MACD","MACD_Signal","OBV"]],data["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

#selecting different models and comparing them
#1. Linear Regression
#2. Random Forest
#3. Logistic Regression
#4. Gradient boosting machines
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}


#parsing data in machine learning model
for model_name, model in models.items():
    
    
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    if model_name in ["Linear Regression","Logistic Regression"]:  # or any other regression model
        # Convert continuous predictions into Buy/Sell/Hold labels based on thresholds
        predictions = [1 if p > 0.5 else -1 if p < -0.5 else 0 for p in predictions]
    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    results[model_name] = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1": f1
    }
    print(f"\nModel: {model_name}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    all_predictions = model.predict(X)
    if model_name == "Linear Regression" or model_name == "Logistic Regression":
        all_predictions = [1 if p > 0.5 else -1 if p < -0.5 else 0 for p in all_predictions]
    
    # Initialize variables
    initial_balance = 100000  # Starting portfolio balance in USD
    balance = initial_balance
    shares = 0  # Track number of shares held
    trade_log = []  # To log each trade's outcome

    # Loop through predictions to simulate trading
    for i in range(0, len(all_predictions)):
        if all_predictions[i] == 1 and shares == 0:  # Buy logic (only buy if not holding stocks)
            buy_price = data['Adj Close'].iloc[i]
            shares = balance / buy_price
            balance = balance - (shares*buy_price)
            trade_log.append(f"Bought at {buy_price}, at {i}")

        elif all_predictions[i] == -1 and shares > 0:  # Sell if the model predicts -1 (sell signal)
            sell_price = data['Adj Close'].iloc[i]
            balance = balance + (shares * sell_price)  # Sell all shares
            shares = 0  # Reset shares to zero after selling
            trade_log.append(f"Sold at {sell_price}, at {i}")


    # Final portfolio value after all trades
    final_balance = balance + shares * data['Adj Close'].iloc[-1]  # If holding shares, calculate their value
    total_return = ((final_balance - initial_balance) / initial_balance) * 100

    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_balance}")
    print(f"Total Return: {total_return:.2f}%")

    # Calculate buy-and-hold return
    buy_and_hold_return = (data['Adj Close'].iloc[-1] - data['Adj Close'].iloc[0]) / data['Adj Close'].iloc[0] * 100
    print(f"Buy-and-Hold Return: {buy_and_hold_return:.2f}%")
    '''
    buy_count = (all_predictions == 1).sum()
    sell_count = (all_predictions == -1).sum()
    hold_count = (all_predictions == 0).sum()
    '''