curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"en-es-tiny","text":"Hello world"}' | jq
