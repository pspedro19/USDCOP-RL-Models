---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/analysis/llm_client.py
  - src/analysis/prompt_templates.py
  - src/news_engine/mcp_data_layer.py
  - src/news_engine/mcp_server.py
  - usdcop-trading-dashboard/lib/chat
  - usdcop-trading-dashboard/app/api/analysis/chat/route.ts
---

# Auditoría de seguridad y calidad de IA

## Fronteras de confianza

Noticias, titulares, URLs, documentos, respuestas de proveedor y preguntas del usuario son contenido no
confiable. El texto externo no puede modificar system prompts, políticas, permisos, herramientas, estrategias,
entitlements ni órdenes. El LLM solo interpreta datos; las métricas, citas, señales y controles financieros
deben ser deterministas y verificables.

## Hallazgos

### AI-P0-001 — contexto de noticias sin aislamiento demostrable

`mcp_data_layer.py` construye `prompt_injection_text` concatenando titulares externos y el MCP lo devuelve como
contexto listo para el LLM. Falta un gate automatizado que demuestre que instrucciones como “ignora reglas”,
exfiltración de secretos o tool calls embebidos son tratadas como datos.

### AI-P1-002 — proveedores con capacidades diferentes

Azure dispone de salida estructurada; Anthropic usa extracción de JSON. El fallback debe mantener el mismo
schema, límites, política de seguridad y fail-closed. Un fallback no puede relajar validación.

### AI-P1-003 — logs y costes

Errores de proveedor no deben registrar prompts completos, secretos, contexto con PII o respuestas sensibles.
Coste, tokens, latencia, modelo, versión de prompt y cache hit deben registrarse sin contenido sensible.

### AI-P1-004 — afirmaciones financieras

La salida debe separar hechos, cálculos deterministas, inferencias y opinión del modelo. Nunca debe presentarse
como asesoría personalizada ni convertirse directamente en orden. Citas se anexan de fuentes deterministas,
no se confían al modelo.

## Corpus adversarial mínimo

- instrucciones maliciosas en título, cuerpo, URL y metadatos;
- delimitadores Markdown/XML/JSON y texto Unicode confusable;
- petición de system prompt, variables de entorno o credenciales;
- intento de aprobar estrategia, alterar riesgo o ejecutar orden;
- contenido muy largo, repetitivo o cost-amplification;
- respuesta inválida, parcial, con NaN o schema incorrecto;
- poisoning: múltiples fuentes copiando la misma falsedad;
- fallback entre proveedores durante error/timeout;
- prompt indirecto dentro de contenido recuperado;
- HTML/Markdown con XSS al renderizar respuesta.

## Gate de aprobación

Cero fuga de secretos, cero acción no autorizada, schema válido en 100% de casos, output encoding seguro,
coste máximo por request, timeout/cancelación y evidencia de modelo+prompt versionados. Las evaluaciones deben
ser deterministas donde sea posible y repetirse con un conjunto congelado antes de cada release.

