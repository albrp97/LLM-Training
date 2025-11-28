# diagnose_environment.ps1
# Diagnóstico completo del entorno para experimentos de cuantización Fase 4
# Este script verifica: Windows/WSL, GPU, CUDA, Python, y dependencias

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "   DIAGNÓSTICO DE ENTORNO - FASE 4" -ForegroundColor Cyan
Write-Host "   Advanced Quantization Experiments" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# 1. Verificar Sistema Operativo
Write-Host "`n[1/8] Sistema Operativo" -ForegroundColor Yellow
Write-Host "OS: " -NoNewline
Get-ComputerInfo | Select-Object -ExpandProperty OsName

# 2. Verificar WSL
Write-Host "`n[2/8] WSL (Windows Subsystem for Linux)" -ForegroundColor Yellow
try {
    $wslStatus = wsl --list --verbose 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL está instalado" -ForegroundColor Green
        Write-Host $wslStatus
        
        # Verificar versión de WSL
        if ($wslStatus -match "VERSION\s+2") {
            Write-Host "✅ WSL2 detectado (necesario para GPU)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  WSL1 detectado - se necesita WSL2 para acceso GPU" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ WSL no está instalado" -ForegroundColor Red
        Write-Host "   Instalar con: wsl --install" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error al verificar WSL: $_" -ForegroundColor Red
}

# 3. Verificar GPU desde Windows
Write-Host "`n[3/8] GPU (desde Windows)" -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU NVIDIA detectada" -ForegroundColor Green
        $parts = $gpuInfo -split ','
        Write-Host "   Modelo: $($parts[0].Trim())" -ForegroundColor Cyan
        Write-Host "   Driver: $($parts[1].Trim())" -ForegroundColor Cyan
        Write-Host "   VRAM Total: $($parts[2].Trim())" -ForegroundColor Cyan
        
        # Extraer VRAM en GB
        $vramMB = [int]($parts[2].Trim() -replace '[^0-9]', '')
        $vramGB = [math]::Round($vramMB / 1024, 1)
        
        if ($vramGB -ge 8) {
            Write-Host "   ✅ VRAM suficiente para modelos de 1B (mínimo 4GB, tienes ${vramGB}GB)" -ForegroundColor Green
        } elseif ($vramGB -ge 4) {
            Write-Host "   ⚠️  VRAM mínima (${vramGB}GB) - algunos métodos pueden fallar" -ForegroundColor Yellow
        } else {
            Write-Host "   ❌ VRAM insuficiente (${vramGB}GB) - se necesitan al menos 4GB" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ No se detectó GPU NVIDIA o drivers no instalados" -ForegroundColor Red
        Write-Host "   Instalar drivers desde: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error al verificar GPU: $_" -ForegroundColor Red
}

# 4. Verificar CUDA Toolkit en Windows
Write-Host "`n[4/8] CUDA Toolkit (Windows)" -ForegroundColor Yellow
try {
    $nvccVersion = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ CUDA Toolkit instalado" -ForegroundColor Green
        $cudaVersion = ($nvccVersion | Select-String -Pattern "release (\d+\.\d+)").Matches.Groups[1].Value
        Write-Host "   Versión: CUDA $cudaVersion" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  CUDA Toolkit no encontrado en PATH" -ForegroundColor Yellow
        Write-Host "   Puede estar instalado pero no configurado en PATH" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  nvcc no encontrado - CUDA Toolkit puede no estar instalado" -ForegroundColor Yellow
    Write-Host "   Descargar de: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# 5. Verificar GPU desde WSL
Write-Host "`n[5/8] GPU desde WSL" -ForegroundColor Yellow
try {
    $wslGpu = wsl nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU accesible desde WSL" -ForegroundColor Green
        Write-Host "   $wslGpu" -ForegroundColor Cyan
    } else {
        Write-Host "❌ GPU no accesible desde WSL" -ForegroundColor Red
        Write-Host "   Solución: Instalar NVIDIA CUDA on WSL" -ForegroundColor Yellow
        Write-Host "   1. Actualizar WSL: wsl --update" -ForegroundColor Yellow
        Write-Host "   2. Instalar CUDA en WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error al verificar GPU en WSL: $_" -ForegroundColor Red
}

# 6. Verificar Python en Windows
Write-Host "`n[6/8] Python (Windows)" -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python instalado" -ForegroundColor Green
        Write-Host "   $pythonVersion" -ForegroundColor Cyan
        
        # Verificar si es Python 3.10+
        $version = ($pythonVersion -replace 'Python ', '')
        if ([version]$version -ge [version]"3.10.0") {
            Write-Host "   ✅ Versión compatible (necesario Python 3.10+)" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Versión antigua - se recomienda Python 3.10+ o 3.12" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Python no encontrado" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error al verificar Python: $_" -ForegroundColor Red
}

# 7. Verificar PyTorch y CUDA
Write-Host "`n[7/8] PyTorch y CUDA (Windows)" -ForegroundColor Yellow
try {
    $torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PyTorch instalado y configurado" -ForegroundColor Green
        $torchCheck -split "`n" | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }
        
        if ($torchCheck -match "CUDA available: True") {
            Write-Host "   ✅ PyTorch puede usar GPU" -ForegroundColor Green
        } else {
            Write-Host "   ❌ PyTorch no puede usar GPU - reinstalar con CUDA" -ForegroundColor Red
            Write-Host "   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  PyTorch no instalado" -ForegroundColor Yellow
        Write-Host "   Instalar con: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Error al verificar PyTorch: $_" -ForegroundColor Yellow
}

# 8. Verificar espacio en disco
Write-Host "`n[8/8] Espacio en Disco" -ForegroundColor Yellow
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)
$totalGB = [math]::Round(($drive.Used + $drive.Free) / 1GB, 2)
Write-Host "   Disco C: ${freeGB}GB libres de ${totalGB}GB" -ForegroundColor Cyan

if ($freeGB -ge 50) {
    Write-Host "   ✅ Espacio suficiente (se necesitan ~20-30GB para experimentos)" -ForegroundColor Green
} elseif ($freeGB -ge 20) {
    Write-Host "   ⚠️  Espacio limitado (${freeGB}GB) - monitorear durante experimentos" -ForegroundColor Yellow
} else {
    Write-Host "   ❌ Espacio insuficiente (${freeGB}GB) - liberar espacio antes de experimentar" -ForegroundColor Red
}

# Resumen Final
Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "   RESUMEN" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

Write-Host "`nRequisitos para Fase 4:" -ForegroundColor Yellow
Write-Host "  [?] WSL2 instalado y funcionando" -ForegroundColor White
Write-Host "  [?] GPU NVIDIA con 8GB+ VRAM" -ForegroundColor White
Write-Host "  [?] GPU accesible desde WSL (nvidia-smi funciona)" -ForegroundColor White
Write-Host "  [?] Python 3.10+ instalado" -ForegroundColor White
Write-Host "  [?] PyTorch con CUDA instalado" -ForegroundColor White
Write-Host "  [?] 20GB+ espacio en disco libre" -ForegroundColor White

Write-Host "`nSi todos los checks son ✅, ejecutar:" -ForegroundColor Green
Write-Host "  cd experiments/advanced-quantization" -ForegroundColor Cyan
Write-Host "  wsl bash verify_env.sh" -ForegroundColor Cyan
Write-Host "`nPara instalar dependencias en WSL:" -ForegroundColor Green
Write-Host "  wsl bash -c 'cd experiments/advanced-quantization && pip install -r requirements.txt'" -ForegroundColor Cyan

Write-Host "`n===========================================" -ForegroundColor Cyan
