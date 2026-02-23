import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { Star, Plus, X, TrendingUp, TrendingDown, Play } from 'lucide-react';
import { stockAPI, POPULAR_STOCKS } from '../services/api';
import { formatUSDToINR } from '../utils/currencyConverter';
import SymbolAutocomplete from '../components/SymbolAutocomplete';
import { hftApiService } from '../services/hftApiService';
import toast from 'react-hot-toast';
import { useAuth } from '../contexts/AuthContext';
import { getUserStorage } from '../utils/userStorage';

const WatchListPage = () => {
  const { user } = useAuth();

  // Start empty — will be populated from user-scoped storage once user is known
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [storageLoaded, setStorageLoaded] = useState(false);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [startingBot, setStartingBot] = useState<string | null>(null);

  // Load watchlist from the correct user-scoped key whenever user identity is resolved
  useEffect(() => {
    if (!user?.username) return; // Wait until user is known
    const userStorage = getUserStorage(user.username);
    const saved = userStorage.getItem('watchlist');
    setWatchlist(saved ? JSON.parse(saved) : []);
    setStorageLoaded(true);
  }, [user?.username]);

  const handleAddToWatchlist = (symbol: string) => {
    const normalized = symbol.trim().toUpperCase();
    if (normalized && !watchlist.includes(normalized)) {
      setWatchlist([...watchlist, normalized]);
      setNewSymbol('');
    }
  };

  // Save watchlist to user-scoped storage and refresh data
  useEffect(() => {
    if (!user?.username || !storageLoaded) return; // Don't save until initial load is done
    const userStorage = getUserStorage(user.username);
    userStorage.setItem('watchlist', JSON.stringify(watchlist));
    if (watchlist.length > 0) {
      loadWatchlistData();
    }
  }, [watchlist, user?.username, storageLoaded]);

  const loadWatchlistData = async () => {
    setLoading(true);
    try {
      const response = await stockAPI.predict(watchlist, 'intraday');

      // Check for errors in metadata
      if (response.metadata?.error) {
        throw new Error(response.metadata.error);
      }

      if (response.predictions) {
        // Filter out predictions with errors
        const validPredictions = response.predictions.filter((p: any) => !p.error);
        setPredictions(validPredictions);
      } else {
        setPredictions([]);
      }
    } catch (error: any) {
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  const removeFromWatchlist = (symbol: string) => {
    setWatchlist(watchlist.filter((s) => s !== symbol));
  };

  const handleStartBot = async (symbol: string) => {
    try {
      setStartingBot(symbol);
      toast.loading(`Starting bot for ${symbol}...`, { id: `start-bot-${symbol}` });

      const result = await hftApiService.startBotWithSymbol(symbol);

      if (result.status === 'pending') {
        toast.success(`Bot initialization started for ${symbol}. Please wait...`, { id: `start-bot-${symbol}`, duration: 3000 });
        // Navigate to BOT page to see progress
        setTimeout(() => {
          window.location.href = '/hft';
        }, 2000);
      } else {
        toast.success(`Bot started successfully for ${symbol}!`, { id: `start-bot-${symbol}` });
        // Navigate to BOT page after a short delay
        setTimeout(() => {
          window.location.href = '/hft';
        }, 1500);
      }
    } catch (error: any) {
      console.error('Error starting bot:', error);
      const errorMsg = error?.response?.data?.detail || error?.message || `Failed to start bot for ${symbol}`;
      toast.error(errorMsg, { id: `start-bot-${symbol}`, duration: 5000 });
    } finally {
      setStartingBot(null);
    }
  };

  return (
    <Layout>
      <div className="space-y-4">
        <div>
          <h1 className="text-xl font-bold text-white mb-1">Watch List</h1>
          <p className="text-gray-400 text-xs">Monitor your favorite stocks</p>
        </div>

        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center gap-2 mb-3">
            <div className="flex-1">
              <SymbolAutocomplete
                value={newSymbol}
                onChange={setNewSymbol}
                onSelect={handleAddToWatchlist}
                placeholder="e.g., AAPL, TCS.NS"
                excludeSymbols={watchlist}
                className="px-3 py-1.5 text-sm"
              />
            </div>
            <button
              onClick={() => handleAddToWatchlist(newSymbol)}
              className="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-sm font-semibold rounded transition-colors flex items-center gap-1.5"
            >
              <Plus className="w-4 h-4" />
              <span>Add</span>
            </button>
          </div>

          <div className="mb-4">
            <p className="text-gray-300 mb-2 text-sm">Quick Add:</p>
            <div className="flex flex-wrap gap-2">
              {POPULAR_STOCKS.slice(0, 10).map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => {
                    if (!watchlist.includes(symbol)) {
                      setWatchlist([...watchlist, symbol]);
                    }
                  }}
                  disabled={watchlist.includes(symbol)}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors ${watchlist.includes(symbol)
                    ? 'bg-slate-600 text-gray-400 cursor-not-allowed'
                    : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                    }`}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-400">Loading...</div>
        ) : predictions.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {predictions.map((pred, index) => {
              const symbol = watchlist[index];
              return (
                <div key={symbol} className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                      <h3 className="text-xl font-bold text-white">{symbol}</h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleStartBot(symbol)}
                        disabled={startingBot === symbol}
                        className="px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded transition-colors flex items-center gap-1.5"
                        title="Start bot with this symbol"
                      >
                        <Play className="w-3 h-3" />
                        {startingBot === symbol ? 'Starting...' : 'Start Bot'}
                      </button>
                      <button
                        onClick={() => removeFromWatchlist(symbol)}
                        className="p-1 hover:bg-slate-700 rounded transition-colors"
                      >
                        <X className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Price</span>
                      <span className="text-white font-semibold">
                        {formatUSDToINR(pred.predicted_price || pred.current_price || 0, pred.symbol)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Action</span>
                      <div className="flex items-center space-x-2">
                        {(pred.action === 'LONG' || pred.action === 'BUY') ? (
                          <TrendingUp className="w-4 h-4 text-green-400" />
                        ) : (pred.action === 'SHORT' || pred.action === 'SELL') ? (
                          <TrendingDown className="w-4 h-4 text-red-400" />
                        ) : null}
                        <span
                          className={`font-semibold ${pred.action === 'LONG' || pred.action === 'BUY'
                            ? 'text-green-400'
                            : pred.action === 'SHORT' || pred.action === 'SELL'
                              ? 'text-red-400'
                              : 'text-yellow-400'
                            }`}
                        >
                          {pred.action === 'LONG' ? 'BUY' : pred.action === 'SHORT' ? 'SELL' : pred.action || 'HOLD'}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Confidence</span>
                      <span className="text-white font-semibold">
                        {((pred.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    {pred.predicted_return !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="text-gray-400">Return</span>
                        <span className={`font-semibold ${pred.predicted_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {pred.predicted_return > 0 ? '+' : ''}{(pred.predicted_return || 0).toFixed(2)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">No stocks in watchlist</div>
        )}
      </div>
    </Layout>
  );
};

export default WatchListPage;

