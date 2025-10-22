curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"en-es-local","text":"Hello world"}' | jq
