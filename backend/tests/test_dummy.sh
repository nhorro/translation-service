curl http://localhost:8080/health
curl -X POST http://localhost:8080/translate \
     -H "Content-Type: application/json" \
     -d '{"model":"dummy","text":"hola mundo"}'
