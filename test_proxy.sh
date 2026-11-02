#!/bin/bash
curl -s -w "\nHTTP: %{http_code}\n" -X POST http://localhost:5173/api/chat/graph \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"你好\"}" | head -c 200
