#!/bin/bash
echo "Health: $(curl -s http://localhost:8000/api/health)"
echo "First request:"
curl -s -w "\nTime: %{time_total}s\n" -X POST http://localhost:8000/api/chat/graph \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}' | head -c 200
echo
echo "Second request:"
curl -s -w "\nTime: %{time_total}s\n" -X POST http://localhost:8000/api/chat/graph \
  -H "Content-Type: application/json" \
  -d '{"message":"空调安装费多少"}' | head -c 200
