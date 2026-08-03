---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---

# Page snapshot

```yaml
- generic [ref=e1]:
  - generic [ref=e2]:
    - banner [ref=e3]:
      - link "GlobalMarkets — inicio" [ref=e4] [cursor=pointer]:
        - /url: /
        - img [ref=e6]
        - generic [ref=e9]: GlobalMarkets
      - link "Iniciar sesión" [ref=e10] [cursor=pointer]:
        - /url: /login
      - link "Crear cuenta" [ref=e11] [cursor=pointer]:
        - /url: /register
    - main [ref=e12]:
      - generic [ref=e13]:
        - generic [ref=e14]:
          - generic [ref=e15]:
            - img [ref=e17]
            - generic [ref=e20]:
              - heading "Inicia sesión" [level=1] [ref=e21]
              - paragraph [ref=e22]: Terminal de trading cuantitativo · acceso para cuentas aprobadas
          - generic [ref=e23]:
            - generic [ref=e24]: Usuario o correo
            - textbox "Usuario o correo" [active] [ref=e25]:
              - /placeholder: usuario o tu@correo.com
          - generic [ref=e26]:
            - generic [ref=e27]: Contraseña
            - generic [ref=e28]:
              - textbox "Contraseña Mostrar contraseña" [ref=e29]:
                - /placeholder: ••••••••
              - button "Mostrar contraseña" [ref=e30]:
                - img [ref=e31]
          - generic [ref=e34]:
            - generic [ref=e35]:
              - img [ref=e36]
              - text: Verificación
              - generic [ref=e39]: ¿Cuánto es 7 + 6?
            - generic [ref=e40]:
              - textbox "Respuesta de verificación" [ref=e41]:
                - /placeholder: respuesta
              - button "Generar nueva operación" [ref=e42]:
                - img [ref=e43]
          - button "Entrar a la terminal" [disabled] [ref=e48]:
            - img [ref=e49]
            - text: Entrar a la terminal
          - generic [ref=e52]:
            - paragraph [ref=e53]:
              - text: ¿No tienes cuenta?
              - link "Solicitar acceso" [ref=e54] [cursor=pointer]:
                - /url: /register
            - button "Explorar como invitado" [ref=e55]:
              - img [ref=e56]
              - text: Explorar como invitado
        - paragraph [ref=e59]: Sesión protegida · actividad registrada en audit log
    - contentinfo [ref=e60]:
      - generic [ref=e61]:
        - paragraph [ref=e62]: Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica riesgo de pérdida total del capital.
        - navigation "Legal" [ref=e63]:
          - link "Metodología" [ref=e64] [cursor=pointer]:
            - /url: /metodologia
          - generic [ref=e65]: ·
          - link "Términos" [ref=e66] [cursor=pointer]:
            - /url: /legal/terminos
          - generic [ref=e67]: ·
          - link "Planes" [ref=e68] [cursor=pointer]:
            - /url: /pricing
        - paragraph [ref=e69]: © 2026 GlobalMarkets Terminal
  - alert [ref=e70]
```