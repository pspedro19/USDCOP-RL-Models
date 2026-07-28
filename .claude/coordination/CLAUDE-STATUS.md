# CLAUDE-STATUS — heartbeat del backlog
# ACTUALIZAR CADA <=5 MIN mientras se trabaja (y al inicio/fin de cada BL).
# El otro LLM LEE este archivo al inicio de cada ciclo. Formato: append-only por ciclo,
# la seccion ACTUAL se sobreescribe.

## ACTUAL
timestamp: 2026-07-27T00:00:00-05:00
estado: IDLE            # IDLE | WORKING | BLOCKED | WAITING_ACK | DONE_CYCLE
bl_activos: []          # ej [BL-01, BL-06]
agentes_en_vuelo: 0     # subagentes paralelos ahora (max 10)
archivos_bloqueados: [] # paths que NADIE mas debe tocar este ciclo
necesito_del_otro: []   # ej ["ACK contrato strategy_schema v2", "review BL-16"]
para_review: []         # BLs mios terminados esperando verificacion del otro

## LOG (append, mas reciente arriba)
- 2026-07-27T00:00 INIT protocolo creado; sin trabajo iniciado.
