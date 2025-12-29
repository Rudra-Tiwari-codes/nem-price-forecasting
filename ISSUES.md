# NEM Price Forecasting - Issues and Logic Errors

> **Generated:** 2025-12-29
> **Status:** Documentation of all identified issues

---

## Critical Issues

### 1. Vercel Deployment Failure - fs Module Usage

**File:** `dashboard/app/api/simulation/route.js`  
**Lines:** 2, 12-15, 23-29

```javascript
import fs from 'fs';  // FAILS on Vercel serverless
// ...
if (fs.existsSync(regionPath)) {
    const data = JSON.parse(fs.readFileSync(regionPath, 'utf8'));
```

**Problem:** Node.js `fs` module does not work on Vercel's edge/serverless runtime. All filesystem operations fail, causing 500 errors.

**Fix:** Remove fs dependency entirely, only use GitHub raw URL fetch as data source.

---

### 2. 24-Hour Data Instead of 48-Hour

**File:** `main.py`  
**Line:** 179

```python
# Get recent price data for chart (288 intervals = 24 hours at 5-min intervals)
recent_df = df.tail(288).copy()
```

**Problem:** Dashboard requires 48 hours of data (576 intervals), but only 288 intervals (24 hours) are exported.

**Fix:** Change `288` to `576` and update the comment.

---

### 3. Prices API Fetches Only 100 Data Points

**File:** `dashboard/app/api/prices/route.js`  
**Line:** 159

```javascript
const last100 = uniquePrices.slice(-100);
```

**Problem:** Even if we have more data, only 100 points (~8 hours) are returned. This contradicts the 48-hour requirement.

**Fix:** Change to `slice(-576)` for 48 hours of data.

---

### 4. Dashboard Shows "24 Hours" But Should Show "48 Hours"

**File:** `dashboard/app/page.js`  
**Line:** 168

```javascript
Price History - {selectedRegion} (Last 24 Hours)
```

**Problem:** UI label says "24 Hours" but should say "48 Hours".

**Fix:** Update label text.

---

## GitHub Actions Issues

### 5. Scheduled Workflow Not Running Every 5 Minutes

**File:** `.github/workflows/update_data.yml`  
**Line:** 5

```yaml
schedule:
  - cron: '*/5 * * * *'
```

**Problem:** Despite `*/5` cron, GitHub Actions only guarantees "best effort" for scheduled workflows. In practice, runs are delayed 30-60+ minutes during high load.

**Root Cause:** GitHub deprioritizes scheduled workflows. They state: "Scheduled workflows can be expected to run once every 5-60 minutes."

**Symptoms observed:**
- Runs spaced 1+ hours apart instead of 5 minutes
- Some runs fail (visible X marks in `gh run list`)

**Options:**
1. Accept hourly+ delays (GitHub limitation)
2. Use external scheduler (e.g., cron job, AWS EventBridge) to trigger `workflow_dispatch`
3. Switch to Vercel Cron functions

---

### 6. Workflow Failures Due to Git Conflicts

**Run ID:** 20561330918  
**Status:** Failed

Some workflow runs fail when concurrent pushes create merge conflicts or when AEMO data is temporarily unavailable.

---

## API Logic Issues

### 7. Demo Data Masks Real Errors

**File:** `dashboard/app/api/prices/route.js`  
**Lines:** 218-246

```javascript
function returnDemoData(reason) {
    // Returns fake data instead of propagating errors
```

**Problem:** When real data fails to load, the API returns fake "demo data" with status 200. Users see charts but don't realize data is fabricated.

**Fix:** Return proper error responses (500/503) instead of fake data.

---

### 8. GitHub Raw URL Hardcoded Username

**File:** `dashboard/app/api/prices/route.js`  
**Line:** 114

```javascript
const simUrl = `https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main/...`
```

**Problem:** GitHub username is hardcoded. If repo is forked or transferred, URLs break.

**Fix:** Use environment variable or derive from Vercel's REPO_OWNER.

---

## Configuration Issues

### 9. React Compiler with Next.js 16.1.1

**File:** `dashboard/next.config.mjs`

```javascript
reactCompiler: true,
```

**File:** `dashboard/package.json`

```json
"babel-plugin-react-compiler": "1.0.0",
"next": "16.1.1",
"react": "19.2.3",
```

**Problem:** React Compiler is experimental. May cause hydration mismatches on Vercel. Also using bleeding-edge React 19 and Next.js 16.

**Recommendation:** If deployment issues persist, try disabling `reactCompiler: true`.

---

### 10. Vercel Root Directory Not Specified

**File:** `vercel.json`

```json
{
    "buildCommand": "npm run build",
    "outputDirectory": ".next",
    "installCommand": "npm install",
    "framework": "nextjs"
}
```

**Problem:** vercel.json is in project root, but dashboard is in `dashboard/` subdirectory. Must configure Root Directory in Vercel project settings to `dashboard`.

---

## Minor Issues

### 11. NEMWEB Scraping May Hit Rate Limits

**File:** `dashboard/app/api/prices/route.js`  
**Lines:** 6-29

The API scrapes NEMWEB directly on each request. No caching beyond `revalidate: 300`. High traffic could trigger AEMO rate limits.

---

### 12. update_readme.py Regex Pattern May Not Match

**File:** `update_readme.py`  
**Lines:** 75-82

```python
table_pattern = r'\| Strategy \| Profit \| Charge Cycles \| Discharge Cycles \|.*?\n\n'
```

If README format changes, the regex won't match and updates silently fail.

---

## Summary Table

| # | Issue | Severity | File | Fix Needed |
|---|-------|----------|------|------------|
| 1 | fs module fails on Vercel | Critical | api/simulation/route.js | Remove fs, use fetch only |
| 2 | 24h data instead of 48h | High | main.py | Change 288 to 576 |
| 3 | Only 100 data points | High | api/prices/route.js | Change 100 to 576 |
| 4 | Wrong UI label | Low | page.js | Update text |
| 5 | GitHub cron delays | Medium | update_data.yml | Accept or use external trigger |
| 6 | Workflow conflicts | Medium | update_data.yml | Add retry/conflict handling |
| 7 | Demo data masks errors | Medium | api/prices/route.js | Return proper errors |
| 8 | Hardcoded GitHub username | Low | api/prices/route.js | Use env var |
| 9 | Experimental React Compiler | Medium | next.config.mjs | Disable if issues persist |
| 10 | Root directory config | Critical | Vercel Dashboard | Set to `dashboard` |

---

## Next Steps

1. Fix critical Vercel deployment issues (#1, #10)
2. Update data interval from 24h to 48h (#2, #3, #4)
3. Push all fixes to GitHub
4. Document GitHub Actions limitations (#5)
