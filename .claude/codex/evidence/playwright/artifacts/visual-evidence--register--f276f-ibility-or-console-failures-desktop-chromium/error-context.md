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
        - link "Volver a iniciar sesión" [ref=e20] [cursor=pointer]:
          - /url: /login
          - img [ref=e21]
          - text: Volver a iniciar sesión
        - generic [ref=e23]:
          - generic [ref=e24]:
            - img [ref=e26]
            - generic [ref=e29]:
              - heading "Crear cuenta" [level=1] [ref=e30]
              - paragraph [ref=e31]: Acceso por aprobación — un administrador revisa cada solicitud.
          - generic [ref=e32]:
            - generic [ref=e34]: Nombre
            - textbox "Nombre" [ref=e35]:
              - /placeholder: Tu nombre
          - generic [ref=e36]:
            - generic [ref=e38]: Correo
            - textbox "Correo" [ref=e39]:
              - /placeholder: tucorreo@dominio.com
          - generic [ref=e40]:
            - generic [ref=e42]: Contraseña
            - textbox "Contraseña 8+ caracteres Mayúscula Minúscula Número" [ref=e43]:
              - /placeholder: ••••••••
            - list [ref=e44]:
              - listitem [ref=e45]:
                - img [ref=e46]
                - text: 8+ caracteres
              - listitem [ref=e49]:
                - img [ref=e50]
                - text: Mayúscula
              - listitem [ref=e53]:
                - img [ref=e54]
                - text: Minúscula
              - listitem [ref=e57]:
                - img [ref=e58]
                - text: Número
          - generic [ref=e61]:
            - generic [ref=e63]: Confirmar contraseña
            - textbox "Confirmar contraseña" [ref=e64]:
              - /placeholder: ••••••••
          - generic [ref=e65]:
            - generic [ref=e67]: Verificación ¿Cuánto es 3 + 9?
            - generic [ref=e68]:
              - textbox "Verificación ¿Cuánto es 3 + 9? Generar nueva operación" [ref=e69]:
                - /placeholder: respuesta
              - button "Generar nueva operación" [ref=e70]:
                - img [ref=e71]
          - button "Solicitar acceso" [disabled] [ref=e76]
          - paragraph [ref=e77]:
            - text: ¿Ya tienes cuenta?
            - link "Inicia sesión" [ref=e78] [cursor=pointer]:
              - /url: /login
    - contentinfo [ref=e79]:
      - generic [ref=e80]:
        - paragraph [ref=e81]: Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica riesgo de pérdida total del capital.
        - navigation "Legal" [ref=e82]:
          - link "Metodología" [ref=e83] [cursor=pointer]:
            - /url: /metodologia
          - generic [ref=e84]: ·
          - link "Términos" [ref=e85] [cursor=pointer]:
            - /url: /legal/terminos
          - generic [ref=e86]: ·
          - link "Planes" [ref=e87] [cursor=pointer]:
            - /url: /pricing
        - paragraph [ref=e88]: © 2026 GlobalMarkets Terminal
  - alert [ref=e89]
```