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
- generic [active] [ref=e1]:
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
        - link "Volver a iniciar sesión" [ref=e14] [cursor=pointer]:
          - /url: /login
          - img [ref=e15]
          - text: Volver a iniciar sesión
        - generic [ref=e17]:
          - generic [ref=e18]:
            - img [ref=e20]
            - generic [ref=e23]:
              - heading "Crear cuenta" [level=1] [ref=e24]
              - paragraph [ref=e25]: Acceso por aprobación — un administrador revisa cada solicitud.
          - generic [ref=e26]:
            - generic [ref=e28]: Nombre
            - textbox "Nombre" [ref=e29]:
              - /placeholder: Tu nombre
          - generic [ref=e30]:
            - generic [ref=e32]: Correo
            - textbox "Correo" [ref=e33]:
              - /placeholder: tucorreo@dominio.com
          - generic [ref=e34]:
            - generic [ref=e36]: Contraseña
            - textbox "Contraseña 8+ caracteres Mayúscula Minúscula Número" [ref=e37]:
              - /placeholder: ••••••••
            - list [ref=e38]:
              - listitem [ref=e39]:
                - img [ref=e40]
                - text: 8+ caracteres
              - listitem [ref=e43]:
                - img [ref=e44]
                - text: Mayúscula
              - listitem [ref=e47]:
                - img [ref=e48]
                - text: Minúscula
              - listitem [ref=e51]:
                - img [ref=e52]
                - text: Número
          - generic [ref=e55]:
            - generic [ref=e57]: Confirmar contraseña
            - textbox "Confirmar contraseña" [ref=e58]:
              - /placeholder: ••••••••
          - generic [ref=e59]:
            - generic [ref=e61]: Verificación ¿Cuánto es 5 + 5?
            - generic [ref=e62]:
              - textbox "Verificación ¿Cuánto es 5 + 5? Generar nueva operación" [ref=e63]:
                - /placeholder: respuesta
              - button "Generar nueva operación" [ref=e64]:
                - img [ref=e65]
          - button "Solicitar acceso" [disabled] [ref=e70]
          - paragraph [ref=e71]:
            - text: ¿Ya tienes cuenta?
            - link "Inicia sesión" [ref=e72] [cursor=pointer]:
              - /url: /login
    - contentinfo [ref=e73]:
      - generic [ref=e74]:
        - paragraph [ref=e75]: Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica riesgo de pérdida total del capital.
        - navigation "Legal" [ref=e76]:
          - link "Metodología" [ref=e77] [cursor=pointer]:
            - /url: /metodologia
          - generic [ref=e78]: ·
          - link "Términos" [ref=e79] [cursor=pointer]:
            - /url: /legal/terminos
          - generic [ref=e80]: ·
          - link "Planes" [ref=e81] [cursor=pointer]:
            - /url: /pricing
        - paragraph [ref=e82]: © 2026 GlobalMarkets Terminal
  - alert [ref=e83]
```