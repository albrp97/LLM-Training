@echo off
REM QLoRA Implementation - Key Commands
REM ===================================
REM All commands used during QLoRA implementation for easy execution

echo.
echo 🚀 QLoRA Implementation Commands
echo ===============================

echo.
echo 📂 Git Commands:
echo git checkout -b feature/quantization-methods
echo git add .
echo git commit -m "Implement comprehensive QLoRA support"
echo git push -u origin feature/quantization-methods

echo.
echo 🐍 Python Testing Commands:
echo python validate_qlora.py
echo python Fine-tuning/01_Train.py
echo python Testing/02_TestModels.py "Models/Qwen3-0.6B-openmath_SFT_LoRa256_QLORA_w4_headbf16" --trunc-eval 2
echo python Testing/03_EvaluationOrchestrator.py

echo.
echo 📊 File Management:
echo dir Models\
echo dir Testing\metrics\

echo.
echo ⚙️ QLoRA Configuration:
echo QUANT_METHOD = "QLORA"
echo merge_after_train = True
echo # Reuses existing LoRA parameters: lora_r, lora_alpha, lora_dropout

echo.
echo ✅ QLoRA Implementation Complete!
echo Ready for production use.

pause