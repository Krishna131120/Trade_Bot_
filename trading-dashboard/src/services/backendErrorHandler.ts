/**
 * Backend Error Handler
 * 
 * Handles different types of backend errors and provides appropriate user guidance.
 * Distinguishes between recoverable state issues and fatal backend errors.
 */

export interface BackendError {
  type: 'dependency' | 'runtime' | 'data' | 'model' | 'connection' | 'unknown';
  severity: 'recoverable' | 'fatal';
  message: string;
  userMessage: string;
  suggestedAction: string;
  canRetry: boolean;
  retryDelay?: number; // seconds
}

export class BackendErrorHandler {
  
  /**
   * Analyze and categorize backend errors
   */
  static analyzeError(error: any): BackendError {
    const errorMessage = error.message || error.toString() || 'Unknown error';
    const errorString = errorMessage.toLowerCase();

    // 1. DEPENDENCY ERRORS (Recoverable)
    if (errorString.includes('no data found for') || 
        errorString.includes('data not found') ||
        errorString.includes('no cached data')) {
      return {
        type: 'dependency',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Stock data not found in system',
        suggestedAction: 'Click "Fetch Data" to download stock data from Yahoo Finance',
        canRetry: true,
        retryDelay: 5
      };
    }

    if (errorString.includes('features not found') ||
        errorString.includes('technical indicators') ||
        errorString.includes('feature calculation failed')) {
      return {
        type: 'dependency',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Technical indicators not calculated',
        suggestedAction: 'System will automatically calculate 50+ technical indicators',
        canRetry: true,
        retryDelay: 3
      };
    }

    if (errorString.includes('model not found') ||
        errorString.includes('models not trained') ||
        errorString.includes('training failed')) {
      return {
        type: 'model',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'ML models not trained for this symbol',
        suggestedAction: 'System will automatically train 4 ML models (takes 60-90 seconds)',
        canRetry: true,
        retryDelay: 10
      };
    }

    // 2. RUNTIME ERRORS (Some recoverable, some fatal)
    if (errorString.includes("unsupported operand type(s) for -: 'datetime.datetime' and 'int'")) {
      return {
        type: 'runtime',
        severity: 'fatal',
        message: errorMessage,
        userMessage: 'Backend datetime calculation error',
        suggestedAction: 'This is a known backend issue. Try a different symbol or contact support.',
        canRetry: false
      };
    }

    if (errorString.includes('dimension mismatch') ||
        errorString.includes('feature mismatch') ||
        errorString.includes('shape mismatch')) {
      return {
        type: 'model',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Model compatibility issue detected',
        suggestedAction: 'System will retrain models with correct dimensions',
        canRetry: true,
        retryDelay: 15
      };
    }

    if (errorString.includes('memory') || errorString.includes('out of memory')) {
      return {
        type: 'runtime',
        severity: 'fatal',
        message: errorMessage,
        userMessage: 'System running low on memory',
        suggestedAction: 'Try again in a few minutes or contact support if problem persists',
        canRetry: true,
        retryDelay: 60
      };
    }

    // 3. DATA ERRORS (Usually recoverable)
    if (errorString.includes('empty dataframe') ||
        errorString.includes('no price data') ||
        errorString.includes('insufficient data')) {
      return {
        type: 'data',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Insufficient stock data for analysis',
        suggestedAction: 'Try a different symbol or check if the symbol is valid',
        canRetry: true,
        retryDelay: 5
      };
    }

    if (errorString.includes('yahoo finance') ||
        errorString.includes('data fetch failed') ||
        errorString.includes('connection timeout')) {
      return {
        type: 'data',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Failed to fetch data from Yahoo Finance',
        suggestedAction: 'Check internet connection and try again',
        canRetry: true,
        retryDelay: 10
      };
    }

    // 4. CONNECTION ERRORS (Recoverable)
    if (errorString.includes('connection refused') ||
        errorString.includes('cannot connect') ||
        errorString.includes('backend server') ||
        errorString.includes('network error')) {
      return {
        type: 'connection',
        severity: 'recoverable',
        message: errorMessage,
        userMessage: 'Cannot connect to backend server',
        suggestedAction: 'Check if backend server is running and try again',
        canRetry: true,
        retryDelay: 15
      };
    }

    // 5. UNKNOWN ERRORS (Treat as potentially recoverable)
    return {
      type: 'unknown',
      severity: 'recoverable',
      message: errorMessage,
      userMessage: 'An unexpected error occurred',
      suggestedAction: 'Try again in a few moments. If problem persists, contact support.',
      canRetry: true,
      retryDelay: 10
    };
  }

  /**
   * Get user-friendly error message with guidance
   */
  static getErrorGuidance(error: any): {
    title: string;
    message: string;
    action: string;
    canRetry: boolean;
    severity: 'info' | 'warning' | 'error';
  } {
    const analyzed = this.analyzeError(error);
    
    return {
      title: this.getErrorTitle(analyzed.type),
      message: analyzed.userMessage,
      action: analyzed.suggestedAction,
      canRetry: analyzed.canRetry,
      severity: analyzed.severity === 'fatal' ? 'error' : 
               analyzed.type === 'dependency' ? 'info' : 'warning'
    };
  }

  /**
   * Check if error indicates missing dependencies
   */
  static isMissingDependency(error: any): boolean {
    const analyzed = this.analyzeError(error);
    return analyzed.type === 'dependency';
  }

  /**
   * Check if error is recoverable through retry
   */
  static isRecoverable(error: any): boolean {
    const analyzed = this.analyzeError(error);
    return analyzed.severity === 'recoverable';
  }

  /**
   * Get recommended retry delay
   */
  static getRetryDelay(error: any): number {
    const analyzed = this.analyzeError(error);
    return analyzed.retryDelay || 10;
  }

  /**
   * Get error title based on type
   */
  private static getErrorTitle(type: string): string {
    switch (type) {
      case 'dependency': return 'Missing Dependencies';
      case 'runtime': return 'System Error';
      case 'data': return 'Data Issue';
      case 'model': return 'Model Issue';
      case 'connection': return 'Connection Error';
      default: return 'Error';
    }
  }

  /**
   * Format error for logging
   */
  static formatForLogging(error: any, context: string): {
    timestamp: string;
    context: string;
    error: BackendError;
    stackTrace?: string;
  } {
    return {
      timestamp: new Date().toISOString(),
      context,
      error: this.analyzeError(error),
      stackTrace: error.stack
    };
  }

  /**
   * Determine if error should trigger automatic dependency resolution
   */
  static shouldAutoResolve(error: any): boolean {
    const analyzed = this.analyzeError(error);
    return analyzed.type === 'dependency' && analyzed.canRetry;
  }
}