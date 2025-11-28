# Research Roadmap - Resumen Ejecutivo

## 🎯 Estado Actual

**3 Fases Completadas** | **12 Modelos Evaluados** | **3 Papers Documentados**

### Mejores Modelos por Categoría

| Categoría | Modelo | Métrica Clave | Por Qué |
|-----------|--------|---------------|---------|
| 🏆 **Overall Best** | 4B-SFT (NoQuant) | AbsDiff 9.0, 7.87GB | Único que funciona en TODOS los dominios |
| 🧠 **OOD Reasoning** | 8B-DoRA (NoQuant) | F1 0.8995 | Mejor generalización fuera de dominio |
| 💻 **Consumer GPU** | 8B-QLoRA | 7.6GB training | 62% menos VRAM que LoRA estándar |
| 💰 **Low VRAM Training** | 8B-VeRA | 16.2GB training | 30-41% menos que LoRA/DoRA |

---

## 🚨 Top 10 Gaps Identificados

1. **❌ Escalado intermedio**: Salto de 8B → 70B sin explorar 14B-30B
2. **❌ QAT**: Post-Training Quantization falla, necesitamos Quantization-Aware Training
3. **❌ PEFT avanzado**: AdaLoRA, LoRA+, Q-Adapter sin evaluar
4. **❌ Multi-task**: Solo single-task SFT, falta multi-domain training
5. **❌ Dominios limitados**: Solo Math/QA/MCQ, falta code/chat/domain-specific
6. **❌ Hiperparámetros**: Single-seed, no sweeps, hiperparámetros no optimizados
7. **❌ Inference**: Latency medida pero no optimizada (vLLM, speculative decoding)
8. **❌ Arquitectura única**: Solo Qwen3, falta Llama/Mistral/MoE
9. **❌ Métricas limitadas**: Solo accuracy, falta throughput/cost/calibration
10. **❌ Data efficiency**: Fixed 1000 samples, no few-shot ni active learning

---

## 🗺️ Roadmap Priorizado (6 Meses)

### **FASE 4: Quantización Avanzada** (Meses 1-2) 🔥
**Objetivo**: Resolver problema PTQ y encontrar mejor estrategia de quantización

**Experimentos Clave**:
- ✅ QAT (Quantization-Aware Training) vs PTQ head-to-head
- ✅ SmoothQuant, AWQ, GPTQ implementation y comparison
- ✅ Mixed-precision strategies (diferentes precisiones por capa)

**Entregable**: Paper "Advanced Quantization for Fine-Tuned LLMs"  
**Hardware**: RTX 4090 (24GB) suficiente  
**Impact**: CRÍTICO - PTQ actual es catastrófica para modelos fine-tuned

---

### **FASE 5: Escalado Intermedio** (Meses 3-4) 🔥
**Objetivo**: Llenar gap 8B-70B con modelos 14B

**Experimentos Clave**:
- ✅ Qwen3-14B full benchmark (base, SFT, LoRA, DoRA, VeRA, QLoRA)
- ✅ Scaling laws analysis (1.7B → 4B → 8B → 14B)
- ✅ ¿14B-QLoRA supera a 8B-DoRA? ¿14B-SFT es viable en consumer?

**Entregable**: "Scaling Laws for Quantized PEFT: 1.7B to 14B"  
**Hardware**: 2× RTX 4090 (48GB) o A100 40GB recomendado  
**Impact**: ALTO - Define si escalar vale la pena para consumer GPUs

---

### **FASE 7: Optimización de Inference** (Meses 5-6)
**Objetivo**: Reducir latency sin sacrificar quality

**Experimentos Clave**:
- ✅ KV-cache quantization (4-bit, 8-bit)
- ✅ Speculative decoding (draft: 1.7B, target: 8B-DoRA)
- ✅ Framework comparison (vLLM, TensorRT-LLM, llama.cpp)

**Entregable**: "Production-Ready Inference Optimization"  
**Hardware**: RTX 4090 suficiente  
**Impact**: MEDIO-ALTO - Crítico para deployment en producción

---

## ⚡ Quick Wins (Próximas 2 Semanas)

### **Semana 1: Validación y Métricas**
1. **Multi-seed validation** (2 días)
   - 3 seeds de 4B-SFT-OpenMath
   - Establecer error bars
   
2. **Métricas adicionales** (1 día)
   - Throughput (tokens/sec)
   - Cost-per-token estimates
   
3. **Hyperparameter sweep limitado** (2 días)
   - Learning rates: 1e-5, 5e-5, 1e-4
   - Solo en 4B-SFT-OpenMath

### **Semana 2: Proof-of-Concept FASE 4**
4. **QAT implementation básica** (3 días)
   - Fork `01_Train.py` → `01_Train_QAT.py`
   - Integrar Hugging Face QAT
   
5. **QAT vs PTQ experiment** (2 días)
   - 4B-QAT-OpenMath vs 4B-PTQ-OpenMath
   - Si QAT funciona → full FASE 4

---

## 📊 Decisiones Críticas

### **Pregunta 1: ¿Qué fase priorizar?**

**Opción A: FASE 4 (Quantización)** ⭐ RECOMENDADO
- ✅ Mayor impacto inmediato
- ✅ PTQ demostró ser problemática
- ✅ Hardware actual suficiente
- ✅ Resultados aplicables a TODAS las escalas

**Opción B: FASE 5 (Escalado 14B)**
- ⚠️ Requiere más hardware (48GB+)
- ⚠️ Puede no agregar mucho vs 8B
- ✅ Completa la curva de escalado
- ✅ Valida patterns en tamaños mayores

**Recomendación**: **FASE 4 primero**, luego FASE 5 si hardware lo permite

---

### **Pregunta 2: ¿Expansión o Profundización?**

**Profundizar (RECOMENDADO para paper de calidad)**:
- Resolver gaps en configuraciones actuales
- Multi-seed, sweeps, optimización
- Mejor caracterización de trade-offs

**Expandir (mejor para cobertura)**:
- Nuevos datasets (code, chat)
- Nuevas arquitecturas (Llama, Mistral)
- Multi-task experiments

**Recomendación**: **70% Profundizar, 30% Expandir**

---

### **Pregunta 3: ¿Target de Publicación?**

**Opción A: Top Conference (ICML, NeurIPS, ICLR)**
- Requiere: 6-8 meses trabajo, resultados muy sólidos
- Necesita: Multi-seed, ablations completas, scaling laws
- Fases necesarias: 4 + 5 + validación exhaustiva

**Opción B: Workshop / Technical Report**
- Requiere: 3-4 meses trabajo
- Menos riguroso pero más rápido
- Fases necesarias: 4 o 5 (una de las dos)

**Opción C: Blog Posts + Open Research**
- Publicar hallazgos continuamente
- Community engagement
- Fases: Iterativo, cada fase = post

**Recomendación**: **Opción C + apuntar a Workshop** (más impacto práctico)

---

## 🎯 Acción Inmediata (Hoy/Mañana)

### **HOY**
```bash
# 1. Revisar roadmap y decidir prioridad
# 2. Setup multi-seed experiment
cp Fine-tuning/01_Train.py experiments/01_Train_MultiSeed.py
# Modificar para iterar seeds 42, 43, 44
```

### **ESTA SEMANA**
```bash
# 3. Implementar QAT básica
# 4. Ejecutar 4B-QAT-OpenMath
# 5. Comparar vs 4B-PTQ-OpenMath
```

### **PRÓXIMAS 2 SEMANAS**
```bash
# 6. Si QAT funciona → commit a FASE 4 completa
# 7. Si QAT falla → considerar FASE 5 (14B)
# 8. Implementar métricas adicionales (throughput, cost)
```

---

## 📈 KPIs de Éxito

### **Objetivos Técnicos (3 meses)**
- [ ] Método de quantización que preserve >95% accuracy con <50% VRAM
- [ ] Demostrar clara ventaja (o diminishing returns) de 14B vs 8B
- [ ] Lograr ≥2× speedup en inference vs baseline actual

### **Objetivos de Investigación (6 meses)**
- [ ] 2 papers/reports publicados
- [ ] Principios generalizables más allá de Qwen3
- [ ] Recetas claras para practitioners

### **Objetivos de Impacto (12 meses)**
- [ ] Fine-tuning de calidad en GPUs <$2000
- [ ] Citaciones / uso en comunidad
- [ ] Contribuciones upstream a PEFT/Transformers

---

## 💡 Ideas Exploratorias (Moonshots)

Si tiempo/recursos sobran:

1. **LoRA Surgery**: ¿Transfer adapters entre tamaños de modelo?
2. **QA-LoRA**: Combinar QAT con PEFT training
3. **Dynamic Rank**: Diferentes ranks por capa (auto-search)
4. **Hybrid Precision**: Batch (4-bit) vs Online (8-bit) serving

---

## 🤝 Recursos Necesarios

| Item | Mínimo | Óptimo | Ideal |
|------|--------|--------|-------|
| **GPU** | 1× RTX 4090 24GB | 2× RTX 4090 48GB | 1× A100 40GB |
| **Permite** | Hasta 8B-QLoRA | 14B-QLoRA | 14B-SFT, 32B-QLoRA |
| **Tiempo** | 6 meses (1 persona) | 3-4 meses (2 personas) | 2-3 meses (team) |
| **Costo aprox.** | $0 (ya tienes) | $1500-2000 | $20k/año cloud |

---

## ✅ Recomendación Final

**Próximos 2-3 meses**: 
1. ✅ **FASE 4 (Quantización)** - Mayor impacto, hardware actual suficiente
2. ✅ Validación multi-seed de resultados actuales
3. ✅ QAT proof-of-concept esta semana

**Meses 4-6**: 
1. ⚡ **FASE 5 (14B)** si hardware permite, o
2. ⚡ **FASE 7 (Inference)** si quieren deployment focus

**Publicación**:
- Target: Workshop paper en 4 meses
- Continuous blog posts
- Open-source todos los scripts

**ROI**: Maximiza impacto científico Y valor práctico para la comunidad 🚀

---

**Siguiente reunión**: Decidir FASE 4 vs FASE 5, discutir hardware upgrade

