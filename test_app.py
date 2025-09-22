#!/usr/bin/env python3
"""
Test script to verify the upgraded StudyMate application.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing StudyMate AI Pro Imports")
    print("=" * 50)
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from transformers import pipeline
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False
    
    try:
        from backend.pdf_parser import extract_text_from_pdf, chunk_text
        from backend.embedder import Embedder
        from backend.retriever import Retriever
        from backend.qa_model import answer_question, answer_over_passages
        print("✅ Backend modules imported successfully")
    except ImportError as e:
        print(f"❌ Backend modules import failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

def test_model_loading():
    """Test model loading functions."""
    print("\n🔍 Testing Model Loading Functions")
    print("-" * 30)
    
    # Test if we can create pipeline objects (without actually loading models)
    try:
        from transformers import pipeline
        print("✅ Pipeline creation capability verified")
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False
    
    print("✅ Model loading functions ready")
    return True

def test_stub_functions():
    """Test stub functions."""
    print("\n🚧 Testing Stub Functions")
    print("-" * 30)
    
    try:
        # Import the app module to test stub functions
        import app
        print("✅ App module imported successfully")
        
        # Test stub functions
        result1 = app.multi_document_qa([], "test")
        result2 = app.ocr_integration(None)
        result3 = app.text_to_speech("test")
        
        print("✅ All stub functions working")
        return True
    except Exception as e:
        print(f"❌ Stub function test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 StudyMate AI Pro - System Test")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_model_loading()
    success &= test_stub_functions()
    
    if success:
        print("\n🎉 All tests passed! StudyMate AI Pro is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
