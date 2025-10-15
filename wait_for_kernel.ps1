# Script de monitoring autonome du kernel Kaggle
# Usage: .\wait_for_kernel.ps1

$ErrorActionPreference = "Continue"
Write-Host "⏳ Monitoring Kaggle kernels..." -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop monitoring (kernel will continue on Kaggle)" -ForegroundColor Yellow
Write-Host ""

$checkInterval = 45  # seconds between checks
$maxChecks = 80      # 80 × 45s = 1 hour max
$checkCount = 0

while ($checkCount -lt $maxChecks) {
    $checkCount++
    $elapsed = $checkCount * $checkInterval
    
    Write-Host "[Check $checkCount / $maxChecks, T+${elapsed}s]" -ForegroundColor Green
    
    # Get most recent kernel
    $kernelList = kaggle kernels list --mine --page-size 1 2>&1 | Out-String
    
    if ($kernelList -match "([a-z0-9-]+)\s+Elonmj\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})") {
        $kernelId = "elonmj/$($Matches[1])"
        $lastRun = $Matches[2]
        
        Write-Host "  Latest kernel: $kernelId" -ForegroundColor White
        Write-Host "  Last run: $lastRun" -ForegroundColor Gray
        
        # Check status
        $status = kaggle kernels status $kernelId 2>&1 | Out-String
        
        if ($status -match "KernelWorkerStatus\.(\w+)") {
            $statusValue = $Matches[1]
            Write-Host "  Status: $statusValue" -ForegroundColor $(if ($statusValue -eq "COMPLETE") { "Green" } elseif ($statusValue -eq "ERROR") { "Red" } else { "Yellow" })
            
            if ($statusValue -eq "COMPLETE") {
                Write-Host ""
                Write-Host "✅ KERNEL COMPLETED!" -ForegroundColor Green
                Write-Host "   Downloading outputs..." -ForegroundColor Cyan
                
                $outputDir = "validation_output/results/$($Matches[1])"
                New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
                kaggle kernels output $kernelId -p $outputDir
                
                Write-Host ""
                Write-Host "📊 Running microscopic analysis..." -ForegroundColor Cyan
                python analyze_microscopic_logs.py $outputDir
                
                Write-Host ""
                Write-Host "✅ ANALYSIS COMPLETE!" -ForegroundColor Green
                exit 0
            }
            elseif ($statusValue -eq "ERROR") {
                Write-Host ""
                Write-Host "❌ KERNEL ERROR!" -ForegroundColor Red
                exit 1
            }
        }
    }
    
    if ($checkCount -lt $maxChecks) {
        Write-Host "  Waiting ${checkInterval}s for next check..." -ForegroundColor Gray
        Write-Host ""
        Start-Sleep -Seconds $checkInterval
    }
}

Write-Host ""
Write-Host "⏱️ Timeout reached (1 hour). Check Kaggle manually." -ForegroundColor Yellow
exit 2
