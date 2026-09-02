# Auditoría final · Laboratorio 5 · Reto Babel

Fecha de verificación: 1 de septiembre de 2026.

## Resultado congelado

| Evidencia | Resultado |
|---|---:|
| Frases oficiales correctas | 240/240 |
| Tokens oficiales correctos | 1,519/1,519 |
| Frases secretas coherentes | 10/10 |
| Parámetros entrenables | 98,810 |
| Frases sin posición | 114/240 |
| Páginas del informe | 7 |
| Diapositivas HTML | 10 |

El artefacto `artefactos/modelo_transformer_optimizado.pt` tiene SHA-256
`4e9287f33300d1bcc94e2489e6da33655d84a8bc8d1588a005ce7f12acda4ffe`.

## Controles ejecutados

- Los tres notebooks se ejecutaron completamente y no contienen salidas de tipo `error` ni celdas de código sin número de ejecución.
- El notebook oficial conserva los bloques 0–7, las comprobaciones base y el entrenamiento histórico. La optimización se añadió como evidencia posterior.
- El modelo guardado se reconstruyó desde `state_dict`, configuración y vocabularios; el conteo resultante fue 98,810.
- Las diez frases secretas se tradujeron otra vez desde el artefacto y coincidieron con el JSON congelado.
- Los hashes SHA-256 de los tres CSV se conservaron en `resultados/resultados_finales_transformer.json`.
- La prueba secreta tiene `secret_used_for_selection: false`; la lista se evaluó después de seleccionar configuración, semilla y 48 épocas.
- El informe se compiló dos veces con `pdflatex`, produjo exactamente siete páginas y no reportó cajas desbordadas.
- La presentación contiene exactamente diez secciones, tres gráficas embebidas como base64, navegación por teclado y estilos de impresión.
- El README contiene instrucciones de instalación, reproducción, trazabilidad, explicación de métricas y correspondencia con la rúbrica.

## Lectura metodológica

El resultado final es mejor que el baseline en los mismos 240 ejemplos: aumenta 5.83 puntos porcentuales en frases exactas y 1.84 puntos en tokens, reduce 14 errores a cero y no agrega parámetros. La ablación posicional produce una caída de 52.50 puntos por frase, por lo que respalda la hipótesis central del laboratorio.

El 100 % no se interpreta como perfección universal. El vocabulario de validación está cubierto por entrenamiento y la tarea usa una gramática sintética cerrada. La prueba confirmatoria siguiente debe ser una cohorte nueva del docente evaluada una sola vez con el artefacto congelado.
