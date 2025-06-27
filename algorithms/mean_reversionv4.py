import numpy as np

class MeanReversionStrategy:
    def __init__(self, nInst=50, short_window=20, long_window=50, entry_z=1.5, 
                 exit_z=0.5, max_position_pct=0.8, volatility_cap=0.12,
                 min_price=5.0, trend_threshold=0.8):
        self.nInst = nInst
        self.short_window = short_window
        self.long_window = long_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_position_pct = max_position_pct
        self.volatility_cap = volatility_cap
        self.min_price = min_price
        self.trend_threshold = trend_threshold
        self.currentPos = np.zeros(nInst)
        self.lastPrices = np.zeros(nInst)
        
    def get_positions(self, prcSoFar):
        """Calculate positions based on mean reversion strategy"""
        nInst, nt = prcSoFar.shape
        
        # Wait until we have enough data
        min_data_days = max(self.short_window, self.long_window) + 5
        if nt < min_data_days:
            return np.zeros(nInst)
        
        current_prices = prcSoFar[:, -1]
        
        # Calculate indicators
        short_prices = prcSoFar[:, -self.short_window:]
        long_prices = prcSoFar[:, -self.long_window:]
        
        short_ma = np.mean(short_prices, axis=1)
        long_ma = np.mean(long_prices, axis=1)
        long_std = np.std(long_prices, axis=1) + 1e-8
        
        returns = np.diff(short_prices, axis=1) / short_prices[:, :-1]
        volatility = np.std(returns, axis=1) * np.sqrt(252)
        trend_strength = short_ma / long_ma - 1
        z_scores = (current_prices - long_ma) / long_std
        
        # Apply filters
        valid = (
            (current_prices >= self.min_price) &
            (volatility <= self.volatility_cap) &
            (np.abs(trend_strength) < self.trend_threshold)
        )  # Fixed missing parenthesis here
        
        # Calculate target positions
        target_positions = np.zeros(nInst)
        for i in range(nInst):
            if not valid[i]:
                continue
                
            z = z_scores[i]
            price = current_prices[i]
            
            # Exit logic
            if self.currentPos[i] != 0 and abs(z) < self.exit_z:
                target_positions[i] = 0
                continue
                
            # Entry logic
            if abs(z) > self.entry_z:
                size = -z * 4000  # Base sizing
                vol_adj = max(0.2, min(1.0, 0.1 / (volatility[i] + 0.05)))
                dollar_size = size * vol_adj
                dollar_size = np.clip(dollar_size, 
                                    -10000*self.max_position_pct, 
                                    10000*self.max_position_pct)
                target_positions[i] = int(dollar_size / price)
        
        # Enforce position limits
        max_shares = (10000 / current_prices).astype(int)
        target_positions = np.clip(target_positions, -max_shares, max_shares)
        
        return target_positions.astype(int)

# Global variables required by competition
nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    """
    Competition interface function - called daily with price history
    Returns integer positions for each instrument
    """
    global currentPos, nInst  # Added nInst to global declaration
    
    # Optimized parameters from grid search
    best_params = {
        'short_window': 20,
        'long_window': 50,
        'entry_z': 1.5,
        'exit_z': 0.4,
        'max_position_pct': 0.8,
        'volatility_cap': 0.12,
        'min_price': 5.0,
        'trend_threshold': 0.8
    }
    
    # Initialize strategy on first call
    if not hasattr(getMyPosition, 'strategy'):
        getMyPosition.strategy = MeanReversionStrategy(nInst=prcSoFar.shape[0], **best_params)
    
    # Get new positions
    new_positions = getMyPosition.strategy.get_positions(prcSoFar)
    
    # Update current positions
    currentPos = new_positions.copy()
    getMyPosition.strategy.currentPos = new_positions.copy()
    getMyPosition.strategy.lastPrices = prcSoFar[:, -1] if prcSoFar.shape[1] > 0 else np.zeros(prcSoFar.shape[0])
    
    return currentPos