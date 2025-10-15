# 🚀 SSH TUNNEL ACCESS GUIDE - USD/COP Trading System

## 📊 SISTEMA COMPLETAMENTE OPERATIVO

✅ **Dashboard migrado al puerto 5000**
✅ **Airflow mantenido en puerto 8080** 
✅ **15 contenedores Docker HEALTHY**
✅ **92,936 registros de datos disponibles**

---

## 🔧 COMANDOS DE ACCESO INMEDIATO

### 1️⃣ ACCESO BÁSICO - DASHBOARD (Recomendado)
```bash
ssh -L 5000:localhost:5000 GlobalForex@48.216.199.139
```
**Luego abrir:** http://localhost:5000

### 2️⃣ ACCESO ESENCIAL - Dashboard + API + Airflow
```bash
ssh -L 5000:localhost:5000 -L 8000:localhost:8000 -L 8080:localhost:8080 GlobalForex@48.216.199.139
```

### 3️⃣ ACCESO COMPLETO - Todos los servicios
```bash
ssh -L 5000:localhost:5000 \
    -L 8000:localhost:8000 \
    -L 8080:localhost:8080 \
    -L 80:localhost:80 \
    -L 443:localhost:443 \
    -L 5432:localhost:5432 \
    -L 6379:localhost:6379 \
    -L 9000:localhost:9000 \
    -L 9001:localhost:9001 \
    -L 9090:localhost:9090 \
    -L 3002:localhost:3002 \
    -L 5050:localhost:5050 \
    -L 8082:localhost:8082 \
    -L 8083:localhost:8083 \
    -L 8085:localhost:8085 \
    -L 8086:localhost:8086 \
    GlobalForex@48.216.199.139
```

---

## 🌐 URLs DE SERVICIOS (Después del Túnel)

| Servicio | URL Local | Credenciales |
|----------|-----------|--------------|
| **🎯 Trading Dashboard** | http://localhost:5000 | - |
| **📊 Trading API** | http://localhost:8000/docs | - |
| **⚡ Airflow** | http://localhost:8080 | admin / admin123 |
| **📈 Grafana** | http://localhost:3002 | admin / admin123 |
| **💾 pgAdmin** | http://localhost:5050 | admin@trading.com / admin123 |
| **🗄️ MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **📊 Prometheus** | http://localhost:9090 | - |

---

## ⚡ PARA EMPEZAR AHORA:

**1. Ejecuta en tu terminal local:**
```bash
ssh -L 5000:localhost:5000 GlobalForex@48.216.199.139
```

**2. Abre tu navegador en:**
```
http://localhost:5000
```

**✅ Sistema 100% funcional usando SSH tunneling**
