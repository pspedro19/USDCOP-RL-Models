# Atestación de DXY de ICE para EXP-TESIS-RL-01

El dataset de investigación declara `ICE_DXY`. ICE ofrece el índice mediante ICE Connect,
ICE Data API, ICE Data Files o ICE Consolidated History; por tanto, un valor descargado de
otro proveedor no certifica esa identidad.

## Archivo requerido

El operador debe exportar desde un canal autorizado un CSV con exactamente una columna de
fecha (`Date`, `Datetime`, `Time` o `Fecha`) y una columna numérica con los valores diarios
del índice. Las fechas deben ser observaciones diarias sin duplicados. Conservar el archivo
original fuera del repositorio si su licencia lo exige y registrar su SHA-256 en el informe.

## Verificación

```powershell
python scripts/diagnostics/verify_macro_declared_identity.py `
  --clean-path data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet `
  --dxy-reference C:\ruta\autorizada\ice_dxy_daily.csv `
  --dxy-value-column Close `
  --output outputs/thesis-repair/macro_identity_research_v2.json
```

El gate solo pasa si DXY supera la coincidencia definida por el contrato y las otras tres
series siguen coincidiendo. Si el archivo no existe, tiene columnas ambiguas, fechas
duplicadas o discrepancias, el resultado debe permanecer `honoured=false`.

No usar `DTWEXBGS`, Investing, Yahoo, TwelveData ni futuros DX como sustitutos silenciosos:
pueden ser proxies o instrumentos relacionados, pero no prueban identidad ICE DXY.

La identidad oficial usa el índice ICE `DXY` (también aparece como `NYICDX`/`USDX` en la
metodología de ICE). Referencias: [catálogo ICE de índices de moneda](https://developer.ice.com/fixed-income-data-services/catalog/ice-data-indices-currency-indices),
[Currency Indices de ICE](https://www.ice.com/fixed-income-data-services/index-solutions/currency-indices)
y [metodología ICE FX Indexes](https://www.ice.com/publicdocs/data/ICE_FX_Indexes_Methodology.pdf).
