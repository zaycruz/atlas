#!/bin/bash
# Quick smoke test for Atlas CLI

echo "🧪 Atlas CLI Smoke Test"
echo "======================="
echo ""

# Check if Python and dependencies are available
echo "1. Checking Python environment..."
python3 -c "import rich; from atlas_main import cli" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Python environment OK"
else
    echo "   ✗ Python environment issues detected"
    exit 1
fi

# Check syntax compilation
echo ""
echo "2. Checking syntax..."
python3 -m py_compile src/atlas_main/cli.py 2>&1
python3 -m py_compile src/atlas_main/ui.py 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ All files compile successfully"
else
    echo "   ✗ Compilation errors detected"
    exit 1
fi

# Check imports
echo ""
echo "3. Checking imports..."
python3 -c "
from atlas_main.cli import main, _print_help, _handle_status
from atlas_main.ui import ConversationShell
from atlas_main.agent import AtlasAgent
from atlas_main.ollama import OllamaClient
print('   ✓ All imports successful')
" 2>&1

if [ $? -eq 0 ]; then
    echo "   Imports verified"
else
    echo "   ✗ Import errors detected"
    exit 1
fi

echo ""
echo "✅ Smoke test passed!"
echo ""
echo "Next steps:"
echo "  1. Run: python -m atlas_main.cli"
echo "  2. Try typing a message"
echo "  3. Test /help command"
echo "  4. Test /status command"
echo "  5. Test Ctrl+D to exit"
