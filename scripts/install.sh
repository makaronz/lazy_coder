#!/bin/bash

# LazyCoder Installation Script
# Automatic installation and configuration of LazyCoder

set -e

echo "🚀 Starting LazyCoder installation..."

# Check Python version
echo "📋 Checking system requirements..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or newer is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version - OK"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed"
    exit 1
fi

echo "✅ pip3 - OK"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔧 Installing LazyCoder in development mode..."
pip install -e .

# Create directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p uploads
mkdir -p data/chroma

# Configure environment file
echo "⚙️ Configuring environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ .env file created from example"
    echo "⚠️  WARNING: Edit the .env file and add your API keys!"
else
    echo "✅ .env file already exists"
fi

# Check spaCy installation
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Check installation
echo "🧪 Checking installation..."
if python -c "import src; print('✅ Import LazyCoder - OK')" 2>/dev/null; then
    echo "✅ LazyCoder installed successfully!"
else
    echo "❌ Error during LazyCoder import"
    exit 1
fi

# Run tests
echo "🧪 Running basic tests..."
if python -m pytest tests/test_basic_functionality.py::TestConfiguration::test_config_loading -v; then
    echo "✅ Basic tests passed successfully!"
else
    echo "⚠️  Some tests failed - check your configuration"
fi

echo ""
echo "🎉 LazyCoder installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit the .env file and add your API keys:"
echo "   - OPENAI_API_KEY=sk-your-key-here"
echo "   - GITHUB_TOKEN=ghp_your-token-here"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Start LazyCoder:"
echo "   lazycoder start"
echo ""
echo "4. Or test file processing:"
echo "   lazycoder process-file example.py"
echo ""
echo "📚 Documentation: README.md"
echo "🆘 Help: lazycoder --help"
echo ""
echo "🚀 Happy coding with LazyCoder!"