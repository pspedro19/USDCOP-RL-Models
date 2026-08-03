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
      - link "Planes" [ref=e10] [cursor=pointer]:
        - /url: /pricing
      - button "Idioma" [ref=e11]:
        - img [ref=e12]
        - text: ES
      - link "Iniciar sesión" [ref=e16] [cursor=pointer]:
        - /url: /login
      - link "Crear cuenta" [ref=e17] [cursor=pointer]:
        - /url: /register
    - main [ref=e18]:
      - generic [ref=e19]:
        - generic [ref=e20]:
          - generic [ref=e21]:
            - img [ref=e23]
            - generic [ref=e26]:
              - heading "Inicia sesión" [level=1] [ref=e27]
              - paragraph [ref=e28]: Terminal de trading cuantitativo · acceso para cuentas aprobadas
          - generic [ref=e29]:
            - generic [ref=e30]: Usuario o correo
            - textbox "Usuario o correo" [active] [ref=e31]:
              - /placeholder: usuario o tu@correo.com
          - generic [ref=e32]:
            - generic [ref=e33]: Contraseña
            - generic [ref=e34]:
              - textbox "Contraseña Mostrar contraseña" [ref=e35]:
                - /placeholder: ••••••••
              - button "Mostrar contraseña" [ref=e36]:
                - img [ref=e37]
          - generic [ref=e40]:
            - generic [ref=e41]:
              - img [ref=e42]
              - text: Verificación
              - generic [ref=e45]: ¿Cuánto es 2 × 9?
            - generic [ref=e46]:
              - textbox "Respuesta de verificación" [ref=e47]:
                - /placeholder: respuesta
              - button "Generar nueva operación" [ref=e48]:
                - img [ref=e49]
          - button "Entrar a la terminal" [disabled] [ref=e54]:
            - img [ref=e55]
            - text: Entrar a la terminal
          - generic [ref=e58]:
            - paragraph [ref=e59]:
              - text: ¿No tienes cuenta?
              - link "Solicitar acceso" [ref=e60] [cursor=pointer]:
                - /url: /register
            - button "Explorar como invitado" [ref=e61]:
              - img [ref=e62]
              - text: Explorar como invitado
        - paragraph [ref=e65]: Sesión protegida · actividad registrada en audit log
    - contentinfo [ref=e66]:
      - generic [ref=e67]:
        - paragraph [ref=e68]: Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica riesgo de pérdida total del capital.
        - navigation "Legal" [ref=e69]:
          - link "Metodología" [ref=e70] [cursor=pointer]:
            - /url: /metodologia
          - generic [ref=e71]: ·
          - link "Términos" [ref=e72] [cursor=pointer]:
            - /url: /legal/terminos
          - generic [ref=e73]: ·
          - link "Planes" [ref=e74] [cursor=pointer]:
            - /url: /pricing
        - paragraph [ref=e75]: © 2026 GlobalMarkets Terminal
  - alert [ref=e76]
```