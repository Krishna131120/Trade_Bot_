/**
 * Backend Dependency Controller
 * 
 * Controls the execution order required by the ML backend system.
 * The backend is a CLI-based research tool that requires strict execution order:
 * 1. Fetch Data → 2. Calculate Features → 3. Train Models → 4. Predict
 * 
 * This controller ensures the frontend never calls backend operations out of order.
 */

import { stockAPI } from './api';

export interface DependencyStatus {
  dataExists: boolean;
  featuresExist: boolean;
  modelsExist: boolean;
  canPredict: boolean;
  missingSteps: string[];
  dataAge?: number;
}

export interface ExecutionStep {
  step: 'fetch_data' | 'calculate_features' | 'train_models' | 'predict';
  description: string;
  required: boolean;
  completed: boolean;
  error?: string;
}

export class BackendDependencyController {
  private static instance: BackendDependencyController;
  
  private constructor() {}
  
  static getInstance(): BackendDependencyController {
    if (!BackendDependencyController.instance) {
      BackendDependencyController.instance = new BackendDependencyController();
    }
    return BackendDependencyController.instance;
  }

  /**
   * Check what dependencies exist for a symbol
   */
  async checkDependencies(symbol: string, horizon: string = 'intraday'): Promise<DependencyStatus> {
    try {
      // Step 1: Check if data exists by attempting to fetch it
      let dataExists = false;
      let dataAge = undefined;
      
      try {
        const dataCheck = await stockAPI.fetchData([symbol], '2y', false, false);
        const result = dataCheck.results?.[0];
        dataExists = result?.status === 'cached' || result?.status === 'success';
        
        if (dataExists && result?.date_range) {
          const endDate = new Date(result.date_range.end);
          const now = new Date();
          dataAge = Math.floor((now.getTime() - endDate.getTime()) / (1000 * 60 * 60 * 24));
        }
      } catch (error: any) {
        // If error contains "No data found", data definitely doesn't exist
        if (error.message && error.message.includes('No data found for')) {
          dataExists = false;
        }
      }

      // Step 2: Check if features exist (only if data exists)
      let featuresExist = false;
      if (dataExists) {
        try {
          const featuresCheck = await stockAPI.fetchData([symbol], '2y', true, false);
          const result = featuresCheck.results?.[0];
          featuresExist = result?.features?.status === 'loaded' || 
                         result?.features?.status === 'calculated';
        } catch {
          featuresExist = false;
        }
      }

      // Step 3: Check if models exist (only meaningful if features exist)
      // We can't directly check model files, but we can infer from prediction attempts
      let modelsExist = false;
      if (featuresExist) {
        // Models are checked during prediction - we'll assume they need to be trained
        // unless we get a successful prediction
        modelsExist = false; // Conservative assumption
      }

      // Determine missing steps
      const missingSteps: string[] = [];
      if (!dataExists) missingSteps.push('fetch_data');
      if (!featuresExist) missingSteps.push('calculate_features');
      if (!modelsExist) missingSteps.push('train_models');

      const canPredict = dataExists && featuresExist; // Models will auto-train if needed

      return {
        dataExists,
        featuresExist,
        modelsExist,
        canPredict,
        missingSteps,
        dataAge
      };
    } catch (error) {
      console.error('[BackendDependencyController] Error checking dependencies:', error);
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
   * Execute the required pipeline to ensure all dependencies are met
   */
  async ensureDependencies(
    symbol: string, 
    horizon: string = 'intraday',
    onProgress?: (step: ExecutionStep) => void
  ): Promise<DependencyStatus> {
    
    const steps: ExecutionStep[] = [
      { step: 'fetch_data', description: 'Fetching stock data', required: true, completed: false },
      { step: 'calculate_features', description: 'Calculating technical indicators', required: true, completed: false },
      { step: 'train_models', description: 'Training ML models', required: true, completed: false }
    ];

    try {
      // Check current status
      const status = await this.checkDependencies(symbol, horizon);
      
      // Step 1: Ensure data exists
      if (!status.dataExists) {
        onProgress?.(steps[0]);
        
        try {
          await stockAPI.fetchData([symbol], '2y', false, true); // force refresh
          steps[0].completed = true;
          onProgress?.(steps[0]);
        } catch (error: any) {
          steps[0].error = error.message;
          steps[0].completed = false;
          onProgress?.(steps[0]);
          throw new Error(`Failed to fetch data: ${error.message}`);
        }
      } else {
        steps[0].completed = true;
      }

      // Step 2: Ensure features exist
      if (!status.featuresExist) {
        onProgress?.(steps[1]);
        
        try {
          await stockAPI.fetchData([symbol], '2y', true, false); // include features
          steps[1].completed = true;
          onProgress?.(steps[1]);
        } catch (error: any) {
          steps[1].error = error.message;
          steps[1].completed = false;
          onProgress?.(steps[1]);
          throw new Error(`Failed to calculate features: ${error.message}`);
        }
      } else {
        steps[1].completed = true;
      }

      // Step 3: Models will be handled by the backend during prediction
      // The MCP adapter automatically trains models if they don't exist
      steps[2].completed = true; // We'll let the backend handle this

      // Return final status
      return await this.checkDependencies(symbol, horizon);
      
    } catch (error: any) {
      console.error('[BackendDependencyController] Pipeline failed:', error);
      throw error;
    }
  }

  /**
   * Validate that prediction can be executed
   */
  async validatePredictionRequest(symbol: string, horizon: string = 'intraday'): Promise<{
    canExecute: boolean;
    reason?: string;
    missingSteps?: string[];
  }> {
    try {
      const status = await this.checkDependencies(symbol, horizon);
      
      if (status.canPredict) {
        return { canExecute: true };
      }
      
      return {
        canExecute: false,
        reason: `Missing dependencies: ${status.missingSteps.join(', ')}`,
        missingSteps: status.missingSteps
      };
    } catch (error: any) {
      return {
        canExecute: false,
        reason: `Dependency check failed: ${error.message}`,
        missingSteps: ['fetch_data', 'calculate_features', 'train_models']
      };
    }
  }

  /**
   * Get user-friendly status message
   */
  getStatusMessage(status: DependencyStatus): string {
    if (status.canPredict) {
      return 'Ready for prediction';
    }
    
    const missing = status.missingSteps;
    if (missing.includes('fetch_data')) {
      return 'No stock data found. Need to fetch data first.';
    }
    if (missing.includes('calculate_features')) {
      return 'Stock data exists but technical indicators not calculated.';
    }
    if (missing.includes('train_models')) {
      return 'Data and features ready. Models will be trained automatically.';
    }
    
    return 'Unknown status';
  }

  /**
   * Get estimated time for missing steps
   */
  getEstimatedTime(missingSteps: string[]): number {
    let totalSeconds = 0;
    
    if (missingSteps.includes('fetch_data')) totalSeconds += 30; // 30 seconds
    if (missingSteps.includes('calculate_features')) totalSeconds += 20; // 20 seconds  
    if (missingSteps.includes('train_models')) totalSeconds += 75; // 75 seconds
    
    return totalSeconds;
  }
}

export const backendDependencyController = BackendDependencyController.getInstance();