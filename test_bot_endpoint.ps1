# Test script for /api/bot/start-with-symbol endpoint

$baseUrl = "http://127.0.0.1:5000"
$endpoint = "/api/bot/start-with-symbol"
$symbol = "RELIANCE.NS"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing BOT Start-With-Symbol Endpoint" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if backend is running
Write-Host "[1] Checking if backend is running..." -ForegroundColor Yellow
try {
    $healthCheck = Invoke-WebRequest -Uri "$baseUrl/api/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Backend is running (Status: $($healthCheck.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend is not running or not accessible on port 5000" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start the backend server first:" -ForegroundColor Yellow
    Write-Host "  cd backend/hft2/backend" -ForegroundColor Yellow
    Write-Host "  python web_backend.py" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 2: Test the start-with-symbol endpoint
Write-Host "[2] Testing POST $endpoint with symbol: $symbol" -ForegroundColor Yellow
try {
    $body = @{
        symbol = $symbol
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "$baseUrl$endpoint" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30 `
        -ErrorAction Stop

    Write-Host "✓ Request successful!" -ForegroundColor Green
    Write-Host "  Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    $responseData = $response.Content | ConvertFrom-Json
    $responseData | ConvertTo-Json -Depth 10 | Write-Host
    
    Write-Host ""
    Write-Host "Summary:" -ForegroundColor Cyan
    Write-Host "  Status: $($responseData.status)" -ForegroundColor $(if ($responseData.status -eq "success") { "Green" } else { "Yellow" })
    Write-Host "  Symbol: $($responseData.symbol)" -ForegroundColor White
    Write-Host "  Bot Running: $($responseData.isRunning)" -ForegroundColor $(if ($responseData.isRunning) { "Green" } else { "Yellow" })
    Write-Host "  Watchlist: $($responseData.watchlist -join ', ')" -ForegroundColor White
    
    if ($responseData.prediction) {
        Write-Host "  Prediction: Available" -ForegroundColor Green
    } else {
        Write-Host "  Prediction: Not available (MCP may not be configured)" -ForegroundColor Yellow
    }
    
    if ($responseData.analysis) {
        Write-Host "  Analysis: Available" -ForegroundColor Green
        Write-Host "    Recommendation: $($responseData.analysis.recommendation)" -ForegroundColor White
        Write-Host "    Confidence: $($responseData.analysis.confidence)" -ForegroundColor White
    } else {
        Write-Host "  Analysis: Not available (MCP may not be configured)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "✗ Request failed!" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "  Status Code: $statusCode" -ForegroundColor Red
        
        try {
            $errorStream = $_.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($errorStream)
            $errorBody = $reader.ReadToEnd()
            Write-Host "  Error Details: $errorBody" -ForegroundColor Red
        } catch {
            Write-Host "  Could not read error details" -ForegroundColor Red
        }
    }
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test completed!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
