/**
 * UI State Controller
 * 
 * Controls UI behavior based on backend dependency status.
 * Prevents users from triggering invalid operations and provides clear guidance.
 */

import { DependencyStatus } from './backendDependencyController';
import { BackendErrorHandler } from './backendErrorHandler';

export interface UIState {
  // Button states
  canPredict: boolean;
  canFetchData: boolean;
  canCalculateFeatures: boolean;
  canTrainModels: boolean;
  
  // Loading states
  isLoading: boolean;
  loadingStep?: string;
  progress?: number;
  
  // Status display
  statusMessage: string;
  statusType: 'info' | 'warning' | 'error' | 'success';
  
  // User guidance
  nextAction?: string;
  estimatedTime?: number;
  
  // Error state
  hasError: boolean;
  errorMessage?: string;
  canRetry?: boolean;
  suggestedAction?: string;
}

export class UIStateController {
  
  /**
   * Calculate UI state based on dependency status
   */
  static calculateUIState(
    symbol: string,
    dependencyStatus: DependencyStatus,
    isLoading: boolean = false,
    loadingStep?: string,
    progress?: number,
    error?: any
  ): UIState {
    
    // Handle error state
    if (error) {
      const guidance = BackendErrorHandler.getErrorGuidance(error);
      return {
        canPredict: false,
        canFetchData: !isLoading,
        canCalculateFeatures: false,
        canTrainModels: false,
        isLoading,
        loadingStep,
        progress,
        statusMessage: guidance.message,
        statusType: guidance.severity,
        nextAction: guidance.action,
        hasError: true,
        errorMessage: guidance.message,
        canRetry: guidance.canRetry,
        suggestedAction: guidance.action
      };\n    }\n\n    // Handle loading state\n    if (isLoading) {\n      return {\n        canPredict: false,\n        canFetchData: false,\n        canCalculateFeatures: false,\n        canTrainModels: false,\n        isLoading: true,\n        loadingStep,\n        progress,\n        statusMessage: loadingStep || 'Processing...',\n        statusType: 'info',\n        hasError: false\n      };\n    }\n\n    // Calculate button states based on dependencies\n    const canPredict = dependencyStatus.canPredict;\n    const canFetchData = true; // Always allow data fetching\n    const canCalculateFeatures = dependencyStatus.dataExists;\n    const canTrainModels = dependencyStatus.dataExists && dependencyStatus.featuresExist;\n\n    // Determine status message and next action\n    let statusMessage: string;\n    let statusType: UIState['statusType'];\n    let nextAction: string | undefined;\n    let estimatedTime: number | undefined;\n\n    if (canPredict) {\n      statusMessage = `Ready to predict ${symbol}`;\n      statusType = 'success';\n      nextAction = 'Click \"Predict\" to generate ML prediction';\n    } else if (!dependencyStatus.dataExists) {\n      statusMessage = `No data found for ${symbol}`;\n      statusType = 'warning';\n      nextAction = 'Click \"Fetch Data\" to download stock data';\n      estimatedTime = 30; // seconds\n    } else if (!dependencyStatus.featuresExist) {\n      statusMessage = `Data exists but technical indicators not calculated`;\n      statusType = 'info';\n      nextAction = 'System will calculate 50+ technical indicators automatically';\n      estimatedTime = 20; // seconds\n    } else {\n      statusMessage = `Data and features ready. Models will train automatically if needed.`;\n      statusType = 'info';\n      nextAction = 'Click \"Predict\" to start prediction (may take 60-90 seconds for first prediction)';\n      estimatedTime = 75; // seconds\n    }\n\n    return {\n      canPredict,\n      canFetchData,\n      canCalculateFeatures,\n      canTrainModels,\n      isLoading: false,\n      statusMessage,\n      statusType,\n      nextAction,\n      estimatedTime,\n      hasError: false\n    };\n  }\n\n  /**\n   * Get button tooltip based on state\n   */\n  static getButtonTooltip(action: 'predict' | 'fetchData' | 'calculateFeatures' | 'trainModels', uiState: UIState): string {\n    switch (action) {\n      case 'predict':\n        if (uiState.canPredict) {\n          return 'Generate ML prediction using 4 trained models';\n        } else {\n          return `Cannot predict: ${uiState.nextAction || 'Missing dependencies'}`;\n        }\n      \n      case 'fetchData':\n        if (uiState.canFetchData) {\n          return 'Download latest stock data from Yahoo Finance';\n        } else {\n          return 'Data fetching in progress...';\n        }\n      \n      case 'calculateFeatures':\n        if (uiState.canCalculateFeatures) {\n          return 'Calculate 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)';\n        } else {\n          return 'Need to fetch stock data first';\n        }\n      \n      case 'trainModels':\n        if (uiState.canTrainModels) {\n          return 'Train 4 ML models: Random Forest, LightGBM, XGBoost, DQN Agent';\n        } else {\n          return 'Need data and features before training models';\n        }\n      \n      default:\n        return '';\n    }\n  }\n\n  /**\n   * Get progress bar configuration\n   */\n  static getProgressConfig(uiState: UIState): {\n    show: boolean;\n    value: number;\n    label: string;\n    color: 'primary' | 'secondary' | 'success' | 'warning' | 'error';\n  } {\n    if (!uiState.isLoading) {\n      return {\n        show: false,\n        value: 0,\n        label: '',\n        color: 'primary'\n      };\n    }\n\n    const progress = uiState.progress || 0;\n    let color: 'primary' | 'secondary' | 'success' | 'warning' | 'error' = 'primary';\n    \n    if (uiState.hasError) {\n      color = 'error';\n    } else if (progress >= 100) {\n      color = 'success';\n    } else if (progress >= 80) {\n      color = 'primary';\n    }\n\n    return {\n      show: true,\n      value: progress,\n      label: uiState.loadingStep || 'Processing...',\n      color\n    };\n  }\n\n  /**\n   * Get status indicator configuration\n   */\n  static getStatusIndicator(uiState: UIState): {\n    icon: string;\n    color: string;\n    message: string;\n    showDetails: boolean;\n  } {\n    if (uiState.hasError) {\n      return {\n        icon: '⚠️',\n        color: 'error',\n        message: uiState.errorMessage || 'Error occurred',\n        showDetails: true\n      };\n    }\n\n    if (uiState.isLoading) {\n      return {\n        icon: '⏳',\n        color: 'info',\n        message: uiState.loadingStep || 'Processing...',\n        showDetails: true\n      };\n    }\n\n    switch (uiState.statusType) {\n      case 'success':\n        return {\n          icon: '✅',\n          color: 'success',\n          message: uiState.statusMessage,\n          showDetails: false\n        };\n      \n      case 'warning':\n        return {\n          icon: '⚠️',\n          color: 'warning',\n          message: uiState.statusMessage,\n          showDetails: true\n        };\n      \n      case 'error':\n        return {\n          icon: '❌',\n          color: 'error',\n          message: uiState.statusMessage,\n          showDetails: true\n        };\n      \n      default:\n        return {\n          icon: 'ℹ️',\n          color: 'info',\n          message: uiState.statusMessage,\n          showDetails: true\n        };\n    }\n  }\n\n  /**\n   * Get estimated time display\n   */\n  static getTimeEstimate(estimatedSeconds: number): string {\n    if (estimatedSeconds < 60) {\n      return `~${estimatedSeconds} seconds`;\n    } else {\n      const minutes = Math.ceil(estimatedSeconds / 60);\n      return `~${minutes} minute${minutes > 1 ? 's' : ''}`;\n    }\n  }\n\n  /**\n   * Check if user should be warned about long operation\n   */\n  static shouldWarnAboutTime(estimatedSeconds: number): boolean {\n    return estimatedSeconds > 45; // Warn if operation takes more than 45 seconds\n  }\n\n  /**\n   * Get dependency checklist for display\n   */\n  static getDependencyChecklist(dependencyStatus: DependencyStatus): Array<{\n    step: string;\n    description: string;\n    completed: boolean;\n    required: boolean;\n  }> {\n    return [\n      {\n        step: 'Data',\n        description: 'Stock price data from Yahoo Finance',\n        completed: dependencyStatus.dataExists,\n        required: true\n      },\n      {\n        step: 'Features',\n        description: '50+ technical indicators (RSI, MACD, etc.)',\n        completed: dependencyStatus.featuresExist,\n        required: true\n      },\n      {\n        step: 'Models',\n        description: '4 ML models (RF, LightGBM, XGBoost, DQN)',\n        completed: dependencyStatus.modelsExist,\n        required: true\n      }\n    ];\n  }\n}