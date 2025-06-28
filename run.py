#!/usr/bin/env python3
"""Startup script for the RAG Systems Comparison Tool."""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_ollama():
    """Check if Ollama is available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is available")
            return True
        else:
            print("❌ Ollama is not running")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama is not installed or not in PATH")
        return False


def check_model():
    """Check if the required model is available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if "gemma3:4b-it-qat" in result.stdout:
            print("✅ Gemma 3 model is available")
            return True
        else:
            print("⚠️  Gemma 3 model not found")
            print("Run: ollama pull gemma3:4b-it-qat")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_docs_folder():
    """Check if docs folder exists and contains PDF files."""
    docs_path = Path("docs")
    if not docs_path.exists():
        print("❌ docs/ folder not found")
        return False
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in docs/ folder")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files in docs/ folder")
    return True


def install_dependencies():
    """Install dependencies using Poetry."""
    try:
        print("📦 Installing dependencies with Poetry...")
        result = subprocess.run(["poetry", "install"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False
    except FileNotFoundError:
        print("❌ Poetry not found. Please install Poetry first:")
        print("curl -sSL https://install.python-poetry.org | python3 -")
        return False


def run_app():
    """Run the Streamlit application."""
    try:
        print("🚀 Starting RAG Systems Comparison Tool...")
        subprocess.run(["poetry", "run", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start the application")
        return False
    except FileNotFoundError:
        print("❌ Poetry not found")
        return False


def main():
    """Main function to check requirements and run the app."""
    print("🤖 RAG Systems Comparison Tool - Startup Check")
    print("=" * 50)
    
    # Check requirements
    checks = [
        ("Python Version", check_python_version),
        ("Ollama Installation", check_ollama),
        ("Gemma 3 Model", check_model),
        ("Documents Folder", check_docs_folder),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if not all_passed:
        print("❌ Some requirements are not met. Please fix the issues above.")
        print("\n📖 For detailed setup instructions, see README.md")
        return
    
    print("✅ All requirements met!")
    
    # Ask user if they want to install dependencies
    install = input("\n📦 Install/update dependencies? (y/N): ").lower().strip()
    if install in ['y', 'yes']:
        if not install_dependencies():
            return
    
    # Ask user if they want to start the app
    start = input("\n🚀 Start the application? (Y/n): ").lower().strip()
    if start not in ['n', 'no']:
        run_app()


if __name__ == "__main__":
    main() 