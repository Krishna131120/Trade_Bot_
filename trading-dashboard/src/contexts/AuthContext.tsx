import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authAPI } from '../services/api';
import { config } from '../config';
import { getUserStorage } from '../utils/userStorage';

interface User {
  username: string;
  token: string;
}

interface AuthContextType {
  user: User | null;
  login: (username: string, password: string) => Promise<void>;
  signup: (username: string, password: string, email: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  _isProvider: boolean; // Internal flag to check if we're inside a provider
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Initialize user from localStorage synchronously to prevent redirect on page reload
const initializeUser = (): User | null => {
  const token = localStorage.getItem('token');
  const username = localStorage.getItem('username');
  // Only return user if token is a valid JWT (not 'no-auth-required')
  if (token && token !== 'no-auth-required' && username) {
    return { username, token };
  }
  // Clear invalid tokens
  if (token === 'no-auth-required') {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
  }
  return null;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Check if backend auth is disabled (open access mode)
  // In open access mode, we allow anonymous users
  const [authEnabled, setAuthEnabled] = useState<boolean | null>(null); // null = not checked yet
  const [user, setUser] = useState<User | null>(initializeUser);

  const checkAuthStatus = useCallback(async () => {
    try {
      const data = await authAPI.checkStatus();

      if (data.auth_status === 'disabled') {
        setAuthEnabled(false);
        const anonymousUser = { username: 'anonymous', token: 'no-auth-required' };
        setUser(anonymousUser);
        localStorage.setItem('token', 'no-auth-required');
        localStorage.setItem('username', 'anonymous');
      } else {
        setAuthEnabled(true);
        const currentToken = localStorage.getItem('token');
        const currentUsername = localStorage.getItem('username');
        if (currentToken === 'no-auth-required') {
          localStorage.removeItem('token');
          localStorage.removeItem('username');
          setUser(null);
        } else if (currentToken && currentUsername && currentToken !== 'no-auth-required') {
          setUser({ username: currentUsername, token: currentToken });
        } else {
          setUser(null);
        }
      }
    } catch (_error) {
      // Backend unreachable or error: require login. Do NOT set anonymous user.
      setAuthEnabled(true);
      setUser(null);
      localStorage.removeItem('token');
      localStorage.removeItem('username');
    }
  }, []);

  useEffect(() => {
    // Check backend auth status on mount
    checkAuthStatus();
  }, [checkAuthStatus]);

  // When auth is enabled and we have no valid user, ensure state is clean (no anonymous)
  useEffect(() => {
    if (authEnabled === true && user?.token === 'no-auth-required') {
      setUser(null);
      localStorage.removeItem('token');
      localStorage.removeItem('username');
    }
  }, [authEnabled, user?.token]);

  const login = useCallback(async (username: string, password: string) => {
    // If auth is disabled, allow any login or skip login
    if (authEnabled === false) {
      const userData = {
        username: username || 'anonymous',
        token: 'no-auth-required'
      };
      setUser(userData);
      localStorage.setItem('token', 'no-auth-required');
      localStorage.setItem('username', userData.username);
      return;
    }

    // Only try to login via API if auth is enabled
    if (authEnabled === true) {
      try {
        const response = await authAPI.login(username, password);

        if (response.success && response.token) {
          const userData = {
            username: response.username || username,
            token: response.token
          };
          setUser(userData);
          localStorage.setItem('token', response.token);
          localStorage.setItem('username', response.username || username);
        } else {
          throw new Error(response.error || 'Login failed');
        }
      } catch (error: any) {
        // Handle axios errors
        if (error.response?.data?.detail) {
          throw new Error(error.response.data.detail);
        }
        throw new Error(error.message || 'Login failed. Please check your credentials.');
      }
    }
  }, [authEnabled]);

  const signup = async (username: string, password: string, email: string) => {
    if (!authEnabled) {
      await login(username, password);
      return;
    }
    try {
      const response = await authAPI.signup(username, password, email);
      if (!response.success) {
        throw new Error((response as any).message || 'Signup failed');
      }
      const r = response as { token?: string; username?: string };
      if (r.token && r.token !== 'no-auth-required') {
        setUser({ username: r.username || username, token: r.token });
        localStorage.setItem('token', r.token);
        localStorage.setItem('username', r.username || username);
      } else {
        await login(username, password);
      }
    } catch (error: any) {
      throw new Error(error.message || 'Signup failed');
    }
  };

  const logout = () => {
    // Clear this user's scoped data from localStorage before removing auth token
    if (user?.username) {
      const userStorage = getUserStorage(user.username);
      userStorage.clearUserData();
    }
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    // Clear any other auth/session keys that might persist
    try {
      const keysToRemove: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && (k === 'token' || k === 'username' || k.startsWith('auth_') || k.startsWith('session_'))) {
          keysToRemove.push(k);
        }
      }
      keysToRemove.forEach((k) => localStorage.removeItem(k));
    } catch {
      // ignore
    }
  };

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, isAuthenticated: !!user, _isProvider: true }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): Omit<AuthContextType, '_isProvider'> => {
  const context = useContext(AuthContext);
  if (!context || !context._isProvider) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  // Remove internal flag before returning
  const { _isProvider, ...publicContext } = context;
  return publicContext;
};

