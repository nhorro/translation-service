# ensure .env has HF_TOKEN and you've accepted the model license on HF
curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"es-en-nllb600m","text":"hola mundo"}' | jq
