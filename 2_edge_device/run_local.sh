#!/bin/bash
# Script chạy hệ thống điều hướng trên máy tính Local (Mac/PC)
# PBL6 - Smart Blind Navigation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/../.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    PYTHON_CMD="$VENV_PYTHON"
else
    PYTHON_CMD="python3"
fi

echo "================================================================="
echo "   KHỞI ĐỘNG HỆ THỐNG ĐIỀU HƯỚNG NGƯỜI KHIẾM THỊ (PBL6 LOCAL)    "
echo "================================================================="
echo "Sử dụng Python: $PYTHON_CMD"
echo ""

$PYTHON_CMD "$SCRIPT_DIR/main.py" "$@"
