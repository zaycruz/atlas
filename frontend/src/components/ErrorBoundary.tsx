import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-atlas-black flex items-center justify-center p-8">
          <div className="max-w-2xl w-full bg-atlas-green-950/20 border border-atlas-red-400/50 rounded-lg p-8">
            <div className="flex items-center gap-3 mb-6">
              <AlertTriangle size={32} className="text-atlas-red-400" />
              <h1 className="text-2xl font-bold text-atlas-red-400">
                System Error Detected
              </h1>
            </div>

            <div className="mb-6">
              <p className="text-atlas-green-400 mb-4">
                A.T.L.A.S. encountered an unexpected error. This has been logged for analysis.
              </p>

              {this.state.error && (
                <div className="bg-black/50 border border-atlas-green-900 rounded p-4 mb-4">
                  <p className="text-xs font-semibold text-atlas-yellow-400 mb-2">
                    ERROR DETAILS
                  </p>
                  <p className="text-xs text-atlas-red-400 font-mono">
                    {this.state.error.toString()}
                  </p>
                  {this.state.errorInfo && (
                    <details className="mt-3">
                      <summary className="text-xs text-atlas-green-500 cursor-pointer hover:text-atlas-cyan-400">
                        Stack Trace
                      </summary>
                      <pre className="text-xs text-atlas-green-700 mt-2 overflow-auto max-h-64">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </details>
                  )}
                </div>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={this.handleReset}
                className="flex items-center gap-2 px-4 py-2 bg-atlas-cyan-400/20 border border-atlas-cyan-400 rounded hover:bg-atlas-cyan-400/30 transition-colors"
              >
                <RefreshCw size={16} className="text-atlas-cyan-400" />
                <span className="text-sm font-semibold text-atlas-cyan-400">
                  Attempt Recovery
                </span>
              </button>

              <button
                onClick={() => window.location.reload()}
                className="flex items-center gap-2 px-4 py-2 bg-atlas-green-950/30 border border-atlas-green-900 rounded hover:border-atlas-green-500 transition-colors"
              >
                <span className="text-sm font-semibold text-atlas-green-500">
                  Reload Application
                </span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
