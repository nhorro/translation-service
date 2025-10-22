curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"es-en-nllb600m-local","text":"¿Cómo estás?","params":{"num_beams":5}}' | jq
