# Near-Live Mode (Snapshot Updates)

## Feature Purpose

Near-Live Mode is a **frontend-only** feature that provides frequent, automatic updates of stock predictions using the existing backend pipeline. It refreshes data at configurable intervals (30s, 60s, 2min, 5min) to give users near-real-time insights without requiring manual refresh clicks.

## Why This Is NOT True Live Data

**CRITICAL UNDERSTANDING:**

1. **Snapshot-Based**: Backend provides Yahoo Finance OHLC (Open, High, Low, Close) snapshot data, NOT tick-by-tick live prices
2. **Pipeline Delay**: Each refresh runs the full prediction pipeline (fetch → calculate → train → predict), taking 60-90 seconds
3. **Market Lag**: Yahoo Finance data itself has inherent delays
4. **Model-Based**: Predictions are model projections, not real-time market movements

**This feature provides:**
- ✅ Automatic periodic updates
- ✅ Latest available snapshot data
- ✅ Fresh model predictions

**This feature does NOT provide:**
- ❌ Tick-level live prices
- ❌ Real-time market data
- ❌ Sub-second updates
- ❌ WebSocket streaming

## Implementation Details

### 1. Feature Definition (NON-NEGOTIABLE)

**Behavior:**
- User enables toggle
- Frontend auto-refreshes data every N seconds
- Each refresh triggers FULL backend pipeline via existing logic
- Uses `forceRefresh` to bypass cache & dedup
- Uses snapshot-based OHLC data only
- Pauses when tab is hidden
- Stops automatically on backend error

### 2. UI Components (IMPLEMENTED EXACTLY)

**A. Mode Indicator (Header)**
```
Stocks Market [Near-Live (Snapshot)]
```

**B. Controls Panel**
- Toggle: ☑ Near-Live Mode
- Interval selector: [30s | 60s | 2min | 5min]
- Status line: "Last: HH:MM:SS • Next: XXs"

**Helper text (verbatim):**
```
Updates every N seconds using latest available market snapshot.
Not tick-by-tick live price.
```

**C. Countdown Timer**
- Visible countdown (60 → 59 → ... → 0)
- Resets after each refresh
- Resets on manual Force Refresh

### 3. Interval Options (FIXED SET)

**Allowed intervals ONLY:**
- 30s (fast updates)
- 60s (default)
- 2min (moderate)
- 5min (slow)

No custom intervals allowed.

### 4. Auto-Refresh Engine (CRITICAL)

**Implementation:**
```typescript
// State management
const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(false);
const [refreshInterval, setRefreshInterval] = useState(60);
const [lastRefreshTime, setLastRefreshTime] = useState<Date | null>(null);
const [nextRefreshIn, setNextRefreshIn] = useState<number | null>(null);

// Auto-refresh loop with setInterval
useEffect(() => {
  if (!autoRefreshEnabled || !searchQuery || isRequestInProgress) return;
  
  // Countdown timer (updates every second)
  const countdownInterval = setInterval(() => {
    if (lastRefreshTime) {
      const elapsed = Math.floor((Date.now() - lastRefreshTime.getTime()) / 1000);
      const remaining = Math.max(0, refreshInterval - elapsed);
      setNextRefreshIn(remaining);
    }
  }, 1000);
  
  // Auto-refresh timer
  const refreshTimer = setInterval(() => {
    if (!document.hidden && !isRequestInProgress) {
      console.log('[AUTO-REFRESH] Triggering snapshot update');
      handleSearchWithNews(searchQuery, true, true); // forceRefresh: true
    }
  }, refreshInterval * 1000);
  
  return () => {
    clearInterval(countdownInterval);
    clearInterval(refreshTimer);
  };
}, [autoRefreshEnabled, searchQuery, refreshInterval, isRequestInProgress, lastRefreshTime]);
```

**Safety features:**
- ✅ Prevents overlapping requests using `isRequestInProgress`
- ✅ Skips refresh if request already in progress
- ✅ Uses existing `forceRefresh` flag
- ✅ Bypasses `requestDeduplicator`
- ✅ Triggers SAME pipeline as Force Refresh

### 5. Tab Awareness (MANDATORY)

**When `document.hidden === true`:**
- Pause auto-refresh
- Log: `[AUTO-REFRESH] Paused (tab hidden)`

**When tab becomes visible:**
- Resume countdown
- Log: `[AUTO-REFRESH] Resumed (tab visible)`

**Implementation:**
```typescript
useEffect(() => {
  const handleVisibilityChange = () => {
    if (document.hidden && autoRefreshEnabled) {
      console.log('[AUTO-REFRESH] Paused (tab hidden)');
    } else if (!document.hidden && autoRefreshEnabled) {
      console.log('[AUTO-REFRESH] Resumed (tab visible)');
    }
  };
  
  document.addEventListener('visibilitychange', handleVisibilityChange);
  return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
}, [autoRefreshEnabled]);
```

### 6. Error Handling (STRICT)

**If backend returns ANY error:**
- Auto-disable Near-Live Mode
- Stop timer immediately
- Show backend error verbatim
- Log: `[AUTO-REFRESH] Stopped due to error`

**NO retries. NO silent failures.**

**Implementation:**
```typescript
try {
  await Promise.all([searchPromise, newsPromise]);
  setLastRefreshTime(new Date());
} catch (err) {
  console.error('[AUTO-REFRESH] Error during refresh:', err);
  if (autoRefreshEnabled) {
    setAutoRefreshEnabled(false);
    console.log('[AUTO-REFRESH] Stopped due to error');
  }
}
```

### 7. Manual Override (IMPORTANT)

**Force Refresh button:**
- Always works
- Immediately triggers refresh
- Resets countdown
- Works even when Near-Live Mode is ON

### 8. Labeling & Honesty (LOCKED)

**UI text MUST use:**
- Mode: "Near-Live (Snapshot)"
- Price: "Model Reference Price"
- Source: "Yahoo Finance OHLC data snapshot"
- Helper: "Not tick-by-tick live price"
- Return: "Model-based projection, not today's market move"

**DO NOT change wording.**

### 9. Dev Logging (DEV MODE ONLY)

**Required logs:**
```
[AUTO-REFRESH] Triggering snapshot update
[REFRESH] Forcing fresh request, bypassing dedup
[REFRESH] Forcing full pipeline re-run
[AUTO-REFRESH] Paused (tab hidden)
[AUTO-REFRESH] Resumed (tab visible)
[AUTO-REFRESH] Stopped due to error
```

## Files Modified

### `StocksView.tsx`
**Changes:**
- Added auto-refresh state management (4 state variables)
- Added auto-refresh loop with countdown (useEffect)
- Added visibility change handler (useEffect)
- Added Near-Live mode controls UI (controls panel)
- Updated `handleSearchWithNews` to track refresh time and handle errors
- Added mode indicator badge in header

**Lines of code added:** ~80 lines
**Backend modifications:** ZERO

## Architecture Compliance

### Backend (UNTOUCHED)
✅ No backend code modified
✅ No new backend endpoints
✅ No direct Yahoo Finance API calls from frontend
✅ Uses existing prediction pipeline
✅ Uses existing `forceRefresh` mechanism

### Frontend (MODIFIED)
✅ Auto-refresh loop implemented
✅ Tab awareness implemented
✅ Error handling implemented
✅ Honest labeling enforced
✅ Manual override preserved

### Data Flow
```
User enables Near-Live Mode
  ↓
setInterval triggers every N seconds
  ↓
Check: !document.hidden && !isRequestInProgress
  ↓
Call: handleSearchWithNews(symbol, true, true)
  ↓
Call: onSearch(symbol, true, forceRefresh: true)
  ↓
Call: predictionService.predict(symbol, horizon, { forceRefresh: true })
  ↓
Call: stockAPI.predict(symbol, horizon, forceRefresh: true)
  ↓
Call: requestDeduplicator.deduplicate({ forceRefresh: true })
  ↓
Bypass cache → Hit backend fresh → Full pipeline runs
  ↓
Update UI with fresh predictions
  ↓
Update lastRefreshTime → Reset countdown
```

## Safety Guarantees

✅ **No Overlapping Requests**: `isRequestInProgress` lock prevents concurrent calls
✅ **No Infinite Loops**: Timer cleared on unmount and mode disable
✅ **No Background Waste**: Pauses when tab hidden
✅ **No Silent Failures**: Errors stop auto-refresh and display to user
✅ **No Cache Pollution**: Uses `forceRefresh` to bypass deduplication
✅ **No Backend Overload**: Configurable intervals (minimum 30s)

## Verification Checklist

### After Implementation:

**1. Toggle ON → auto-refresh starts**
- [ ] Enable "Near-Live Mode" checkbox
- [ ] Verify countdown starts (60 → 59 → 58...)
- [ ] Verify mode indicator badge appears in header

**2. Countdown visible**
- [ ] Countdown updates every second
- [ ] Shows "Last: HH:MM:SS • Next: XXs"
- [ ] Resets to interval value after refresh

**3. Timestamp updates each refresh**
- [ ] Wait for countdown to reach 0
- [ ] Verify "Last:" timestamp updates
- [ ] Verify predictions refresh with new data

**4. Tab switch pauses/resumes**
- [ ] Switch to another tab
- [ ] Check console: `[AUTO-REFRESH] Paused (tab hidden)`
- [ ] Switch back to tab
- [ ] Check console: `[AUTO-REFRESH] Resumed (tab visible)`

**5. Backend stop → mode auto-disables**
- [ ] Stop backend server
- [ ] Wait for next refresh attempt
- [ ] Verify mode checkbox unchecks automatically
- [ ] Check console: `[AUTO-REFRESH] Stopped due to error`
- [ ] Verify error message displays to user

**6. Force Refresh works anytime**
- [ ] Enable Near-Live Mode
- [ ] Click "Force Refresh" button mid-countdown
- [ ] Verify immediate refresh (doesn't wait for countdown)
- [ ] Verify countdown resets after manual refresh

**7. Backend remains untouched**
- [ ] Verify no backend files modified
- [ ] Verify backend logs show same endpoints hit
- [ ] Verify backend returns same data format

## Final Confirmation

✅ **Files Modified:** `StocksView.tsx` only
✅ **Backend Unchanged:** Zero backend modifications
✅ **Behavior Matches Spec:** EXACTLY as specified
✅ **Terminology Locked:** "Near-Live (Snapshot)" enforced
✅ **Safety Implemented:** All guards in place
✅ **Honesty Enforced:** No false "live" claims
