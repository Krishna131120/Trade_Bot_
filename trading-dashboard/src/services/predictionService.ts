import { stockAPI } from './api';
import { backendDependencyController, DependencyStatus, ExecutionStep } from './backendDependencyController';
import { BackendErrorHandler, BackendError } from './backendErrorHandler';
import type { PredictionItem } from '../types';

// Types
export type PredictStatus = 'success' | 'failed';

export interface PredictOutcome {
  symbol: string;
  status: PredictStatus;
  data?: PredictionItem;
  error?: string;
}

interface ProgressUpdate {
  step: 'checking_dependencies' | 'predicting' | 'complete' | 'error';
  description: string;
  progress: number; // 0-100
  symbol: string;
  error?: string;
  canRetry?: boolean;
  suggestedAction?: string;
}

// Error classes
class PredictionError extends Error {
  constructor(message: string, public code: string, public details?: any) {
    super(message);
    this.name = 'PredictionError';
  }
}

// Frontend must not interpret backend dependency state
// Backend handles all dependency orchestration automatically

export class PredictionService {
  private static instance: PredictionService;
  private listeners: Set<(update: ProgressUpdate) => void> = new Set();

  private constructor() {}

  static getInstance(): PredictionService {
    if (!PredictionService.instance) {
      PredictionService.instance = new PredictionService();
    }
    return PredictionService.instance;
  }

  // Add progress listener
  addProgressListener(listener: (update: ProgressUpdate) => void): void {
    this.listeners.add(listener);
  }

  // Remove progress listener
  removeProgressListener(listener: (update: ProgressUpdate) => void): void {
    this.listeners.delete(listener);
  }

  // Notify all listeners
  private notifyProgress(update: ProgressUpdate): void {
    this.listeners.forEach(listener => listener(update));
  }

  /**
   * Check backend dependencies for a symbol using the dependency controller
   */
  async checkDependencies(symbol: string, horizon: string = 'intraday'): Promise<DependencyStatus> {
    try {
      // First check if backend is reachable
      await stockAPI.health();
      
      // Use the dependency controller to check status
      return await backendDependencyController.checkDependencies(symbol, horizon);
    } catch (error: any) {
      console.error(`[PredictionService] Backend health check failed:`, error);
      const analyzed = BackendErrorHandler.analyzeError(error);
      
      if (analyzed.type === 'connection') {
        throw new PredictionError('Cannot connect to backend server', 'CONNECTION_ERROR');
      }
      
      // Return safe defaults for other errors
      return {
        dataExists: false,
        featuresExist: false,
        modelsExist: false,
        canPredict: false,
        missingSteps: ['fetch_data', 'calculate_features', 'train_models']
      };
    }
  }

  /**
   * Ensure all data dependencies are met before prediction
   */
  async ensureDataDependencies(
    symbol: string,
    horizon: string = 'intraday',
    dependencyStatus?: DependencyStatus
  ): Promise<DependencyStatus> {
    const status = dependencyStatus || await this.checkDependencies(symbol, horizon);

    this.notifyProgress({
      step: 'checking_dependencies',
      description: `Analyzing ${symbol}...`,
      progress: 10,
      symbol
    });

    // If all dependencies are met, return immediately
    if (status.canPredict) {
      this.notifyProgress({
        step: 'checking_dependencies',
        description: `Analyzing ${symbol}...`,
        progress: 30,
        symbol
      });
      return status;
    }

    // Use the dependency controller to ensure all steps are completed
    try {
      const finalStatus = await backendDependencyController.ensureDependencies(
        symbol,
        horizon,
        (step: ExecutionStep) => {
          // Silent execution - show neutral progress only
          let progress = 20;

          switch (step.step) {
            case 'fetch_data':
              progress = step.completed ? 40 : 30;
              break;
            case 'calculate_features':
              progress = step.completed ? 60 : 50;
              break;
            case 'train_models':
              progress = step.completed ? 80 : 70;
              break;
          }

          this.notifyProgress({
            step: 'checking_dependencies',
            description: `Analyzing ${symbol}...`,
            progress,
            symbol,
            error: step.error
          });
        }
      );

      return finalStatus;
    } catch (error: any) {
      const analyzed = BackendErrorHandler.analyzeError(error);
      throw new PredictionError(analyzed.userMessage, 'BACKEND_ERROR');
    }
  }

  /**
   * Execute prediction using /tools/predict only (Market Scan contract)
   */
  async predict(
    symbol: string,
    horizon: 'intraday' | 'short' | 'long' = 'intraday',
    options: {
      riskProfile?: string;
      stopLossPct?: number;
      capitalRiskPct?: number;
      drawdownLimitPct?: number;
      forceRefresh?: boolean;
      _retryCount?: number;
    } = {}
  ): Promise<PredictOutcome> {
    const normalizedSymbol = symbol.trim().toUpperCase();

    // DEV-ONLY: Log force refresh
    if (import.meta.env.DEV && options.forceRefresh) {
      console.log('[REFRESH] Forcing full pipeline re-run for', normalizedSymbol);
    }

    if (import.meta.env.DEV) {
      console.log(`[API] POST /tools/predict called for ${normalizedSymbol}`);
      console.log(`[API] Request payload:`, { symbols: [normalizedSymbol], horizon, forceRefresh: options.forceRefresh || false });
    }

    try {
      this.notifyProgress({
        step: 'checking_dependencies',
        description: options.forceRefresh ? 'Refreshing analysis...' : `Analyzing ${normalizedSymbol}...`,
        progress: 0,
        symbol: normalizedSymbol
      });

      // Execute prediction directly
      this.notifyProgress({
        step: 'predicting',
        description: options.forceRefresh ? `Refreshing prediction for ${normalizedSymbol}...` : `Generating prediction for ${normalizedSymbol}...`,
        progress: 50,
        symbol: normalizedSymbol
      });

      const result = await stockAPI.predict(
        [normalizedSymbol],
        horizon,
        options.riskProfile,
        options.stopLossPct,
        options.capitalRiskPct,
        options.drawdownLimitPct,
        options.forceRefresh || false
      );

      const outcome = this.normalizePredictResponse(result, normalizedSymbol);

      if (outcome.status === 'success') {
        if (import.meta.env.DEV) {
          console.log(`[API] ✅ Success - prediction generated for ${normalizedSymbol}`);
          console.log(`[API] Response data:`, outcome.data);
        }

        this.notifyProgress({
          step: 'complete',
          description: `Prediction complete for ${normalizedSymbol}`,
          progress: 100,
          symbol: normalizedSymbol
        });
      } else {
        if (import.meta.env.DEV) {
          console.log(`[API] ❌ Failed - ${normalizedSymbol}: ${outcome.error || 'Unknown error'}`);
        }
        this.notifyProgress({
          step: 'error',
          description: outcome.error || 'Prediction failed',
          progress: 0,
          symbol: normalizedSymbol,
          error: outcome.error
        });
      }

      return outcome;
    } catch (error: any) {
      const message = error?.message || 'Prediction failed';

      console.error(`[PredictionService] Prediction failed for ${normalizedSymbol}:`, error);

      this.notifyProgress({
        step: 'error',
        description: message,
        progress: 0,
        symbol: normalizedSymbol,
        error: message
      });

      return {
        symbol: normalizedSymbol,
        status: 'failed',
        error: message
      };
    }
  }

  /**
   * Normalize prediction response to the strict Market Scan contract:
   * { results: [{ symbol, status, data?, error? }] }
   */
  private normalizePredictResponse(response: any, requestedSymbol: string): PredictOutcome {
    const fallbackSymbol = requestedSymbol.trim().toUpperCase();

    if (!response || typeof response !== 'object') {
      return {
        symbol: fallbackSymbol,
        status: 'failed',
        error: 'Invalid response from backend'
      };
    }

    // Primary contract: results array
    if (Array.isArray(response.results)) {
      const match = response.results.find((item: any) => {
        const itemSymbol = typeof item?.symbol === 'string' ? item.symbol.trim().toUpperCase() : '';
        return itemSymbol === fallbackSymbol;
      }) || response.results[0];

      if (!match) {
        return {
          symbol: fallbackSymbol,
          status: 'failed',
          error: 'No result returned for symbol'
        };
      }

      const symbol = typeof match.symbol === 'string' && match.symbol.trim() ? match.symbol : fallbackSymbol;
      const status = String(match.status || '').toLowerCase() === 'success' ? 'success' : 'failed';

      if (status === 'success') {
        if (match.data && typeof match.data === 'object') {
          return {
            symbol,
            status,
            data: { ...(match.data as PredictionItem), symbol }
          };
        }

        return {
          symbol,
          status: 'failed',
          error: match.error || 'Prediction data missing for symbol'
        };
      }

      return {
        symbol,
        status: 'failed',
        error: match.error || 'Prediction unavailable for this symbol'
      };
    }

    // Legacy contract: predictions array
    if (Array.isArray(response.predictions)) {
      const prediction = response.predictions.find((item: any) => {
        const itemSymbol = typeof item?.symbol === 'string' ? item.symbol.trim().toUpperCase() : '';
        return itemSymbol === fallbackSymbol;
      }) || response.predictions[0];

      if (!prediction) {
        return {
          symbol: fallbackSymbol,
          status: 'failed',
          error: 'No prediction returned from backend'
        };
      }

      const symbol = typeof prediction.symbol === 'string' && prediction.symbol.trim() ? prediction.symbol : fallbackSymbol;

      if (prediction.error) {
        return {
          symbol,
          status: 'failed',
          error: prediction.error
        };
      }

      return {
        symbol,
        status: 'success',
        data: { ...(prediction as PredictionItem), symbol }
      };
    }

    return {
      symbol: fallbackSymbol,
      status: 'failed',
      error: 'Invalid response from backend'
    };
  }

  /**
   * Batch predict multiple symbols
   */
  async batchPredict(
    symbols: string[],
    horizon: 'intraday' | 'short' | 'long' = 'intraday',
    options: {
      riskProfile?: string;
      stopLossPct?: number;
      capitalRiskPct?: number;
      drawdownLimitPct?: number;
    } = {}
  ): Promise<Record<string, PredictOutcome>> {
    const results: Record<string, PredictOutcome> = {};

    // Process symbols sequentially to avoid overwhelming the backend
    for (const symbol of symbols) {
      const outcome = await this.predict(symbol, horizon, options);
      results[outcome.symbol] = outcome;
    }

    return results;
  }

  /**
   * Get dependency status for a symbol
   */
  async getDependencyStatus(symbol: string, horizon: string = 'intraday'): Promise<DependencyStatus> {
    return await this.checkDependencies(symbol, horizon);
  }

  /**
   * Get user-friendly status message
   */
  getStatusMessage(status: DependencyStatus): string {
    return backendDependencyController.getStatusMessage(status);
  }

  /**
   * Get estimated time for missing dependencies
   */
  getEstimatedTime(missingSteps: string[]): number {
    return backendDependencyController.getEstimatedTime(missingSteps);
  }

  /**
   * Check if an error indicates missing dependencies that can be auto-resolved
   */
  canAutoResolveError(error: any): boolean {
    return BackendErrorHandler.shouldAutoResolve(error);
  }
}

// Export singleton instance and related types
export const predictionService = PredictionService.getInstance();
export type { DependencyStatus, ExecutionStep, BackendError };
