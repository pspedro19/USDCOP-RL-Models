# 🔑 Configuración de API Key para Datos en Tiempo Real

## ⚠️ Problema Actual
El dashboard está usando la clave "demo" de TwelveData que tiene limitaciones muy estrictas y NO permite obtener datos históricos ni en tiempo real del gap del 22 de agosto al 2 de septiembre.

## ✅ Solución: Obtener API Key GRATUITA

### Paso 1: Obtener tu API Key (10 segundos)
1. Ve a: **https://twelvedata.com/pricing**
2. Haz clic en "Get Started" o "Sign Up"
3. Regístrate con tu email
4. Tu API key aparecerá inmediatamente
5. **Es GRATIS para siempre** (hasta 800 llamadas/día)

### Paso 2: Configurar el Dashboard
1. Abre el archivo `.env.local` en la carpeta del dashboard
2. Cambia la línea:
   ```
   TWELVE_DATA_API_KEY=demo
   ```
   Por:
   ```
   TWELVE_DATA_API_KEY=tu_clave_api_aqui
   ```

### Paso 3: Reiniciar el Servidor
```bash
# Detén el servidor con Ctrl+C
# Luego reinicia:
npm run dev
```

## 📊 Qué Obtendrás con la API Key Válida

✅ **Datos históricos reales** del 22 agosto - 2 septiembre
✅ **Datos en tiempo real** cada 5 minutos
✅ **800 llamadas diarias** (más que suficiente)
✅ **Sin limitaciones** de la clave "demo"
✅ **Gratis para siempre**

## 🚀 Verificar que Funciona

1. Abre el dashboard: http://localhost:3000
2. Haz clic en "Align Dataset"
3. Deberías ver:
   - Descarga de datos históricos faltantes
   - Actualización en tiempo real cada 5 minutos
   - Sin errores de API

## ❌ Si Sigues Viendo Errores

1. Verifica que copiaste bien la API key
2. Asegúrate de reiniciar el servidor
3. Revisa la consola del navegador (F12) para más detalles

## 📝 Notas Importantes

- **NO uses la clave "demo"** - tiene límites muy bajos
- La API key gratuita incluye **800 llamadas/día**
- Los datos se actualizan cada **5 minutos** durante horario de mercado
- Horario de mercado: Lunes-Viernes, 8:00 AM - 12:55 PM (hora Colombia)

---

💡 **Tip**: Guarda tu API key en un lugar seguro para futuros proyectos