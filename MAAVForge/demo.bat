@echo off
echo ========================================================
echo   KLA Semiconductor Image Restoration - Pipeline Demo
echo ========================================================
echo.

echo [1/4] Cleaning previous output directories...
if exist "results\test_predictions" rmdir /s /q "results\test_predictions"
if exist "results\side_by_side" rmdir /s /q "results\side_by_side"
if exist "Chip_Test\Output" rmdir /s /q "Chip_Test\Output"
echo Done.
echo.

echo [2/4] Running Official Hackathon Benchmarking Script (evaluate.py)...
echo Model: NAFNet-SR (4.26M Params)
echo Feature: 8-Fold Test-Time Augmentation
python evaluate.py --input_dir Dataset\NoisyLR --output_dir results\test_predictions --model_weights checkpoints\best_model.pt
echo.

echo [3/4] Generating Side-by-Side Visual Proofs (visualize_test.py)...
python visualize_test.py --noisy_dir Dataset\NoisyLR --pred_dir results\test_predictions --output_dir results\side_by_side
echo.

echo [4/4] Running Real-World Custom Chip Data (process_custom.py)...
python process_custom.py
echo.

echo ========================================================
echo   PIPELINE COMPLETE! 
echo   Check results/side_by_side and Chip_Test/Output
echo ========================================================
pause
