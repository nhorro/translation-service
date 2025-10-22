curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{ "model":"es-en-tiny", "text":"hola mundo", "params":{"max_new_tokens":64} }' | jq
