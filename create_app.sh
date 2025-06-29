#!/bin/bash

# Create macOS .app bundle for Strange Attractor Math Course

echo "Creating Strange Attractor Math Course.app..."

# Set variables
APP_NAME="Strange Attractor Math Course"
APP_DIR="$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
PROJECT_DIR="$(pwd)"

# Remove old app if exists
if [ -d "$APP_DIR" ]; then
    echo "Removing old app bundle..."
    rm -rf "$APP_DIR"
fi

# Create directory structure
echo "Creating app bundle structure..."
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create the launcher script
echo "Creating launcher script..."
cat > "$MACOS_DIR/StrangeAttractor" << 'EOF'
#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$DIR/../Resources"

# Change to project directory
cd "$RESOURCES_DIR/StrangeAttractorMathCourse"

# Launch the Python GUI
/usr/bin/python3 launch_course.py
EOF

# Make launcher executable
chmod +x "$MACOS_DIR/StrangeAttractor"

# Copy project files to Resources
echo "Copying project files..."
cp -r "$PROJECT_DIR" "$RESOURCES_DIR/StrangeAttractorMathCourse"

# Copy icon
if [ -f "resources/StrangeAttractor.icns" ]; then
    echo "Copying icon..."
    cp "resources/StrangeAttractor.icns" "$RESOURCES_DIR/StrangeAttractor.icns"
fi

# Create Info.plist
echo "Creating Info.plist..."
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Strange Attractor Math Course</string>
    <key>CFBundleDisplayName</key>
    <string>Strange Attractor Math Course</string>
    <key>CFBundleIdentifier</key>
    <string>com.strangeattractor.mathcourse</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleExecutable</key>
    <string>StrangeAttractor</string>
    <key>CFBundleIconFile</key>
    <string>StrangeAttractor</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
EOF

# Clean up unnecessary files
echo "Cleaning up..."
find "$RESOURCES_DIR/StrangeAttractorMathCourse" -name "*.pyc" -delete
find "$RESOURCES_DIR/StrangeAttractorMathCourse" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf "$RESOURCES_DIR/StrangeAttractorMathCourse/.git"
rm -f "$RESOURCES_DIR/StrangeAttractorMathCourse/create_app.sh"
rm -f "$RESOURCES_DIR/StrangeAttractorMathCourse/create_icon.py"

echo ""
echo "✅ App bundle created successfully!"
echo ""
echo "The app is ready at: $APP_DIR"
echo ""
echo "To install:"
echo "1. Drag '$APP_DIR' to your Applications folder"
echo "2. First time running: Right-click and select 'Open' (due to unsigned app)"
echo ""
echo "The app will:"
echo "- Launch Jupyter notebooks"
echo "- Run visualization demos"
echo "- Show the math cheatsheet"
echo "- Open documentation"