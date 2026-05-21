#!/bin/bash

echo "=== Docker Network Debugging Script ==="
echo "Date: $(date)"
echo

echo "1. Testing basic network connectivity..."
ping -c 3 8.8.8.8 && echo "✓ Internet connectivity OK" || echo "✗ Internet connectivity FAILED"
echo

echo "2. Testing DNS resolution..."
nslookup archive.ubuntu.com && echo "✓ DNS resolution OK" || echo "✗ DNS resolution FAILED"
echo

echo "3. Testing Ubuntu package repository access..."
curl -I http://archive.ubuntu.com/ubuntu/ && echo "✓ Repository access OK" || echo "✗ Repository access FAILED"
echo

echo "4. Checking Docker daemon status..."
docker info > /dev/null 2>&1 && echo "✓ Docker daemon OK" || echo "✗ Docker daemon FAILED"
echo

echo "5. Checking Docker networks..."
docker network ls
echo

echo "6. Testing Docker container network connectivity..."
echo "Running network test in container..."
docker run --rm alpine:latest sh -c "ping -c 2 8.8.8.8 && echo '✓ Container network OK'" || echo "✗ Container network FAILED"
echo

echo "7. Checking build context size..."
echo "Current directory size: $(du -sh . | cut -f1)"
echo "Build context size (with .dockerignore):"
docker build --no-cache --progress=plain -f debug-dockerfile -t network-test . 2>&1 | grep "transferring context" || echo "Build context check completed"
echo

echo "8. Testing Docker build with verbose output..."
echo "Building minimal test image..."
docker build --no-cache --progress=plain -f debug-dockerfile -t network-test . 2>&1 | head -20
echo

echo "=== Debugging complete ==="
echo "If all tests pass, the issue might be:"
echo "- Temporary network glitch (retry the build)"
echo "- Large build context causing timeout"
echo "- Docker daemon resource limits"
echo "- Corporate firewall/proxy issues"