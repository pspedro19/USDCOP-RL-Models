# CONTRACTS — log APPEND-ONLY de cambios a contratos compartidos
# Contrato = cualquier archivo bajo src/contracts/, src/core/contracts/,
# usdcop-trading-dashboard/lib/contracts/, esquemas SQL nuevos, o formato de
# artefactos que el otro consume (bundles, status files, JSONs publicados).
#
# PROTOCOLO: 1) ANTES de cambiar: append entrada PROPOSED aqui + status
# necesito_del_otro=["ACK <id>"]. 2) El otro appendea ACK <id> (o OBJECION con
# alternativa) en su siguiente ciclo (<=10 min). 3) Solo con ACK se implementa;
# se appendea APPLIED con commit hash. 4) Si el otro esta IDLE>15min, auto-ACK
# permitido SOLO para cambios ADITIVOS (campos nuevos opcionales) — jamas para
# breaking.
#
# Formato entrada:
# ## C-NNN | PROPOSED|ACK|OBJECION|APPLIED | quien | timestamp
# archivo: <path> · cambio: <1 linea> · breaking: si|no · consumidores: <quien>

(sin entradas)
