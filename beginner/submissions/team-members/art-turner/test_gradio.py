"""Quick test to verify Gradio app loads correctly."""

import os
import sys
from dotenv import load_dotenv

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Gradio app initialization...")
print("=" * 80)

# Check API keys
print("\n1. Checking environment variables...")
if os.getenv("OPENAI_API_KEY"):
    print("   ✓ OPENAI_API_KEY found")
else:
    print("   ✗ OPENAI_API_KEY missing")

if os.getenv("TAVILY_API_KEY"):
    print("   ✓ TAVILY_API_KEY found")
else:
    print("   ✗ TAVILY_API_KEY missing")

# Import and create app
print("\n2. Importing Gradio app...")
try:
    from app import create_app
    print("   ✓ App module imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Create app instance
print("\n3. Creating Gradio app instance...")
try:
    app = create_app()
    print("   ✓ App created successfully")
except Exception as e:
    print(f"   ✗ App creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify app structure
print("\n4. Verifying app structure...")
if hasattr(app, 'blocks'):
    print("   ✓ App has blocks attribute")
else:
    print("   ⚠️  App structure may be unexpected")

print("\n" + "=" * 80)
print("✅ Gradio app test passed!")
print("\nTo launch the app, run:")
print("   python app.py")
print("\nThe app will be available at: http://localhost:7860")
print("=" * 80)
