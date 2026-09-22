#!/usr/bin/env bash
docker build -f Dockerfile.ollama -t sensai-ollama .
docker rm -f ollama 2>/dev/null
docker run -d -p 11434:11434 -v ollama:/root/.ollama --name ollama sensai-ollama
