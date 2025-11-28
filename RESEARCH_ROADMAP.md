# Research Roadmap & Future Experiments

**Fecha**: Noviembre 2025  
**Autores**: Alberto Rodero & Pablo Lobato

---

## 📊 Análisis del Estado Actual

### Trabajo Completado (3 Fases)

#### **Fase 1: Tamaño de Modelo y LoRA Rank** (0.6B-1.7B)
- ✅ Comparación 0.6B vs 1.7B
- ✅ Análisis de rank LoRA (r=32 a r=1024)
- ✅ Full FT vs PEFT trade-offs
- **Hallazgo clave**: Full FT en modelos pequeños > LoRA en modelos grandes para tareas in-domain

#### **Fase 2: Métodos PEFT** (0.6B-1.7B)
- ✅ Comparación exhaustiva LoRA vs DoRA vs VeRA
- ✅ Análisis de eficiencia de VRAM en entrenamiento
- ✅ Cross-task transfer y catastrophic forgetting
- **Hallazgo clave**: VeRA para restricciones extremas; LoRA/DoRA cuando es viable; Full FT cuando es posible

#### **Fase 3: Escalado y Quantización** (4B-8B)
- ✅ Full FT en 4B (baseline de calidad)
- ✅ LoRA/DoRA/VeRA en 8B con QLoRA
- ✅ Post-Training Quantization (4-bit, 8-bit)
- **Hallazgos clave**: 
  - 4B-SFT: mejor opción balanceada
  - 8B-DoRA: mejor para OOD reasoning
  - QLoRA: 62% reducción VRAM, permite 8B en <12GB
  - PTQ: último recurso (catastrófico para numeracy fine-tuned)

### Configuraciones Probadas: 12 Modelos

| Modelo | Tamaño | Training | Quantization | Mejor Uso |
|--------|--------|----------|--------------|-----------|
| Qwen3-1.7B-base | 1.7B | None | None | Baseline entry |
| Qwen3-4B-base | 4B | None | None/Int8/4bit | Baseline mid-tier |
| **Qwen3-4B-SFT** | 4B | **Full FT** | **None** | **⭐ Champion in-domain** |
| Qwen3-8B-base | 8B | None | None | Quality baseline |
| Qwen3-8B-LoRA | 8B | LoRA (r=256) | None/QLoRA | OOD reasoning |
| **Qwen3-8B-DoRA** | 8B | **DoRA (r=256)** | **None** | **⭐ Champion OOD** |
| Qwen3-8B-VeRA | 8B | VeRA (r=512) | None | Min training VRAM |

### Métricas Actuales (Mejores Resultados)

**In-Domain Math (OpenMath AbsDiff, lower is better):**
- 🥇 4B-SFT: **9.0** (0.61s latency, 7.87GB VRAM)
- 🥈 8B-QLoRA: 117.0 (5.40s latency, 6.02GB VRAM)
- 8B-base: 145.0 (baseline sin fine-tuning)

**OOD Reasoning (ARC F1, higher is better):**
- 🥇 8B-DoRA: **0.8995** (2.59s latency)
- 🥈 8B-LoRA: 0.8995 (3.39s latency)
- 4B-SFT: 0.8259

**OOD QA (SQuAD F1, higher is better):**
- 🥇 8B-base: **50.00** (sin fine-tuning)
- 🥈 8B-VeRA: 49.33
- 🥉 8B-LoRA: 49.31

---

## 🎯 Gaps Identificados y Oportunidades

### 1. **Gap de Escala: Falta 14B-30B**
**Problema**: Salto directo de 8B a modelos >70B sin explorar el rango medio
- No hay datos para Qwen3-14B o modelos similares
- Espacio no explorado entre "consumer GPU" y "data center"
- ¿El patrón "4B-SFT > 8B-LoRA in-domain" se mantiene?

### 2. **Gap de Quantización: QAT No Explorada**
**Problema**: Solo PTQ y QLoRA probados
- **QAT (Quantization-Aware Training)** no ha sido evaluada
- PTQ demostró ser catastrófica para modelos fine-tuned
- QLoRA funciona pero con penalizaciones
- **SmoothQuant**, **AWQ**, **GPTQ** tienen stubs pero sin implementación completa

### 3. **Gap de Métodos PEFT Avanzados**
**Problema**: Métodos más nuevos sin evaluar
- **AdaLoRA** (adaptive rank): mencionado pero no probado
- **LoRA+** y variantes recientes
- **Q-Adapter**, **LongLoRA** para contextos largos
- Combinaciones híbridas (DoRA + quantización específica)

### 4. **Gap de Multi-Task Learning**
**Problema**: Solo SFT single-task evaluado
- No hay experimentos de **multi-task training simultáneo**
- Falta evaluación de **adapter routing** para múltiples dominios
- No se probó **continual learning** ni **catastrophic forgetting mitigation**

### 5. **Gap de Datasets y Dominios**
**Problema**: Solo 3 tasks evaluados (ARC, OpenMath, SQuAD)
- Falta evaluación en **código** (HumanEval, MBPP)
- No hay benchmarks de **chat/instruction-following** (MT-Bench, AlpacaEval)
- Ausencia de tareas **multilingües**
- No se probó **domain-specific** (legal, medical, finance)

### 6. **Gap de Optimización e Hiperparámetros**
**Problema**: Single-seed, single-epoch, hiperparámetros fijos
- No hay **sweeps de learning rate**
- Falta experimentar con **schedulers** (cosine, warmup)
- **Batch size** y **gradient accumulation** no optimizados
- Solo 1 época probada (¿overfitting en más épocas?)

### 7. **Gap de Inference Optimization**
**Problema**: Latency medida pero no optimizada
- No se probó **KV-cache quantization**
- Falta **speculative decoding**
- **Flash Attention** no evaluado explícitamente
- No hay comparación con **vLLM**, **TensorRT-LLM**, **llama.cpp**

### 8. **Gap de Arquitectura**
**Problema**: Solo Qwen3 evaluado
- Falta comparación con **Llama 3/3.1/3.2**
- No se probó **Mistral**, **Gemma**, **Phi**
- Modelos **MoE** (Mixtral) no evaluados
- Diferencias arquitecturales no estudiadas

### 9. **Gap de Métricas de Producción**
**Problema**: Solo métricas académicas (F1, AbsDiff)
- Falta **throughput** (tokens/sec, requests/sec)
- No hay **cost-per-token** analysis
- **Calibration** y **uncertainty** no medidos
- Ausencia de métricas de **safety/toxicity**

### 10. **Gap de Data Efficiency**
**Problema**: 1,000 samples fixed
- No se estudió **few-shot** (10, 50, 100 samples)
- Falta análisis de **data quality vs quantity**
- **Active learning** no explorado
- **Synthetic data** no probada

---

## 🗺️ Roadmap de Experimentos Futuros

### **FASE 4: Optimización de Quantización (Prioridad Alta) 🔥**
**Objetivo**: Resolver el problema PTQ y encontrar la mejor estrategia de quantización

#### Experimentos Propuestos:

**4.1. Quantization-Aware Training (QAT)**
```
Configuración:
- Modelos: 4B, 8B
- Método: QAT desde scratch en fine-tuning
- Comparación vs: PTQ, QLoRA, NoQuant
- Métricas: Same as current + convergence speed

Hipótesis: QAT debería superar a PTQ en modelos fine-tuned
Timeline: 2-3 semanas
Hardware: RTX 4090 o similar
```

**4.2. Métodos de Quantización Avanzados**
```
Implementar y comparar:
1. SmoothQuant (activation + weight quantization)
2. AWQ (Activation-aware Weight Quantization)
3. GPTQ (Post-training quantization optimizado)
4. HQQ (Half-Quadratic Quantization) - ya mencionado

Configuración:
- Baseline: 4B-SFT (mejor modelo actual)
- Target: Mantener <3% accuracy drop
- VRAM target: <4GB inference

Timeline: 3-4 semanas
```

**4.3. Mixed-Precision Strategies**
```
Experimentos:
- Quantizar solo ciertas capas (attention vs FFN)
- Head layers en bf16, resto en 4-bit
- Gradual quantization (primeras capas en mayor precisión)

Objetivo: Encontrar sweet spot precision/performance
Timeline: 2 semanas
```

### **FASE 5: Escalado Intermedio (Prioridad Alta) 🔥**
**Objetivo**: Cubrir el gap 8B-70B con modelos intermedios

#### Experimentos Propuestos:

**5.1. Qwen3-14B Benchmark**
```
Configuración completa:
1. 14B-base (baseline)
2. 14B-SFT-NoPEFT (si VRAM permite)
3. 14B-LoRA (r=256)
4. 14B-DoRA (r=256)
5. 14B-VeRA (r=512)
6. 14B-QLoRA (critical path)

Pregunta clave: ¿14B-QLoRA supera a 8B-DoRA in OOD?
¿14B-SFT es viable en 48GB VRAM?

Timeline: 4-5 semanas
Hardware: 2x RTX 4090 o A100 40GB
```

**5.2. Comparative Scaling Laws**
```
Probar múltiples tamaños con same training data:
- 1.7B, 4B, 8B, 14B (y 32B si posible)
- Mismo dataset, mismo # epochs
- Graficar: accuracy vs parameters, accuracy vs VRAM, etc.

Objetivo: Determinar curva de escalado óptima
Timeline: 6 semanas (parallelizable)
```

### **FASE 6: Multi-Task y Modularidad (Prioridad Media)**
**Objetivo**: Evaluar capabilities multi-dominio y adapter routing

#### Experimentos Propuestos:

**6.1. Multi-Task Training**
```
Setup:
- Entrenar UN modelo en Math + QA + Code simultáneamente
- Comparar vs: 3 modelos especializados
- Métodos: NoPEFT, LoRA, DoRA

Métricas:
- Performance en cada tarea
- Total VRAM (3 modelos vs 1 multi-task)
- Serving latency

Timeline: 3 semanas
```

**6.2. Adapter Routing & Composition**
```
Experimentos:
1. Task-specific adapters con routing automático
2. Mixture of LoRA Experts (MoLE)
3. Adapter fusion strategies

Objetivo: ¿Podemos tener múltiples dominios en 1 base model?
Timeline: 4 semanas
```

**6.3. Catastrophic Forgetting Mitigation**
```
Probar técnicas:
1. Elastic Weight Consolidation (EWC)
2. Progressive Neural Networks
3. PackNet
4. Replay buffers

Target: Mantener performance en tarea A al fine-tunear en tarea B
Timeline: 3-4 semanas
```

### **FASE 7: Optimización de Inference (Prioridad Media)**
**Objetivo**: Reducir latency manteniendo quality

#### Experimentos Propuestos:

**7.1. KV-Cache Optimization**
```
Técnicas:
1. KV-cache quantization (4-bit, 8-bit)
2. Multi-Query Attention (MQA) vs Grouped-Query (GQA)
3. PagedAttention (vLLM-style)

Métrica crítica: Latency en batch inference
Timeline: 2-3 semanas
```

**7.2. Speculative Decoding**
```
Setup:
- Draft model: 1.7B
- Target model: 8B-DoRA
- Medir: speedup, accuracy preservation

Objetivo: 2-3× speedup en generación
Timeline: 2 semanas
```

**7.3. Inference Framework Comparison**
```
Benchmark mismo modelo en:
1. Transformers (baseline actual)
2. vLLM
3. TensorRT-LLM
4. llama.cpp
5. ExLlamaV2

Configuración: 4B-SFT, 8B-DoRA
Métricas: throughput, latency, VRAM
Timeline: 1-2 semanas
```

### **FASE 8: Expansión de Datasets (Prioridad Media-Baja)**
**Objetivo**: Evaluar en más dominios

#### Experimentos Propuestos:

**8.1. Code Tasks**
```
Datasets:
1. HumanEval (Python code generation)
2. MBPP (basic programming)
3. CodeContests (competitive programming)

Modelos: 4B-SFT, 8B-DoRA (best current)
Timeline: 2-3 semanas
```

**8.2. Chat/Instruction Following**
```
Benchmarks:
1. MT-Bench (multi-turn conversation)
2. AlpacaEval
3. IFEval (instruction following)

Objetivo: ¿Cómo se comportan nuestros modelos en chat?
Timeline: 2 semanas
```

**8.3. Domain-Specific Applications**
```
Probar 1-2 dominios específicos:
Opción A: Legal (ContractNLI, LegalBench)
Opción B: Medical (MedQA, PubMedQA)
Opción C: Finance (FiQA, FinanceBench)

Setup: Fine-tune 4B en dominio específico
Timeline: 3-4 semanas por dominio
```

### **FASE 9: Data Efficiency (Prioridad Media-Baja)**
**Objetivo**: Reducir cantidad de datos necesarios

#### Experimentos Propuestos:

**9.1. Few-Shot Learning Curve**
```
Entrenar con:
- 10, 50, 100, 500, 1000, 5000 samples
- Mismo modelo (4B)
- Mismas tasks

Graficar: accuracy vs training samples
Objetivo: Determinar minimum viable dataset size
Timeline: 3 semanas
```

**9.2. Data Quality Analysis**
```
Experimentos:
1. Random sampling vs curated sampling
2. Hard examples vs easy examples
3. Data augmentation techniques

Comparar: 1000 random vs 500 curated
Timeline: 2-3 semanas
```

**9.3. Synthetic Data Generation**
```
Usar modelos grandes para generar training data:
- GPT-4 / Claude para generar ejemplos
- Self-distillation
- Curriculum learning con synthetic progression

Timeline: 3-4 semanas
```

### **FASE 10: Cross-Architecture Analysis (Prioridad Baja)**
**Objetivo**: Validar hallazgos en otras arquitecturas

#### Experimentos Propuestos:

**10.1. Llama 3.2 Comparison**
```
Replicar mejores experimentos con Llama:
- Llama-3.2-3B vs Qwen3-4B
- Llama-3.2-8B vs Qwen3-8B
- Same training protocol

Pregunta: ¿Los patrones son architecture-agnostic?
Timeline: 4-5 semanas
```

**10.2. MoE Exploration**
```
Probar con Mixtral u otros MoE:
- Mixtral-8x7B con LoRA/QLoRA
- Comparar vs dense models

Hipótesis: MoE puede tener mejor quality/VRAM ratio
Timeline: 3-4 semanas (si hardware permite)
```

---

## 📋 Priorización y Timeline

### **Roadmap de 6 Meses (Prioridad Alta)**

**Mes 1-2: FASE 4 - Optimización de Quantización**
- Semanas 1-3: QAT implementation y experiments
- Semanas 4-6: SmoothQuant, AWQ, GPTQ comparison
- Semanas 7-8: Mixed-precision strategies
- **Entregable**: Paper "Advanced Quantization Strategies for LLM Fine-Tuning"

**Mes 3-4: FASE 5 - Escalado Intermedio**
- Semanas 9-13: 14B full benchmark suite
- Semanas 14-16: Scaling laws analysis
- **Entregable**: "Scaling Laws for Quantized Fine-Tuning: 1.7B to 14B"

**Mes 5-6: FASE 7 - Inference Optimization**
- Semanas 17-19: KV-cache optimization
- Semanas 20-21: Speculative decoding
- Semanas 22-24: Framework comparison
- **Entregable**: "Production-Ready Inference: Optimizing Latency and Throughput"

### **Roadmap de 12 Meses (Completo)**

**Mes 7-9: FASE 6 - Multi-Task Learning**
- Multi-task training experiments
- Adapter routing implementation
- Catastrophic forgetting mitigation
- **Entregable**: "Modular LLMs: Multi-Domain Adaptation Strategies"

**Mes 10-12: FASE 8 & 9 - Expansión**
- Code tasks evaluation
- Chat benchmarks
- Domain-specific applications
- Data efficiency studies
- **Entregable**: "Comprehensive LLM Fine-Tuning: From Few-Shot to Multi-Domain"

**Opcional (Mes 12+): FASE 10 - Cross-Architecture**
- Llama 3.2 replication
- MoE exploration
- **Entregable**: "Architecture-Agnostic Fine-Tuning Principles"

---

## 🎯 Recomendaciones Inmediatas (Próximas 2 Semanas)

### **Quick Wins (1-2 días cada uno)**

1. **Implementar métricas adicionales**
   - Throughput (tokens/sec)
   - Cost-per-token estimations
   - Memory bandwidth utilization

2. **Multi-seed validation**
   - Rerun 4B-SFT con 3 seeds
   - Calcular mean ± std
   - Validar robustez de conclusiones

3. **Hyperparameter sweep (limitado)**
   - Probar learning rates: 1e-5, 5e-5, 1e-4
   - En 4B-SFT (modelo crítico)
   - Solo OpenMath (task crítica)

### **High-Impact Experiments (1-2 semanas)**

1. **QAT vs PTQ head-to-head**
   - Implementar QAT simple
   - Comparar 4B-QAT vs 4B-PTQ
   - Si QAT funciona → priorizar FASE 4

2. **14B-QLoRA viability test**
   - ¿Cabe en 24GB VRAM?
   - ¿Mejora sobre 8B-DoRA?
   - Si sí → priorizar FASE 5

3. **vLLM inference benchmark**
   - Portar 4B-SFT y 8B-DoRA a vLLM
   - Medir speedup
   - Si >2× → priorizar FASE 7

---

## 🔬 Experimentos Exploratorios (Moonshots)

### **1. LoRA Surgery**
Idea: ¿Se pueden "trasplantar" LoRA adapters entre modelos de diferentes tamaños?
- Entrenar LoRA en 4B-OpenMath
- Aplicar a 8B-base (con scaling apropiado)
- ¿Funciona como transfer learning?

### **2. Quantization-Aware LoRA**
Idea: Combinar QAT con PEFT training
- Entrenar LoRA mientras base model se quantiza gradualmente
- Posiblemente mejor que QLoRA

### **3. Dynamic Rank Allocation**
Idea: Diferentes ranks para diferentes capas
- Capas tempranas: rank bajo
- Capas de output: rank alto
- Auto-search con NAS

### **4. Hybrid Precision Deployment**
Idea: Diferentes quantizaciones para batch vs online serving
- Batch: 4-bit (throughput priority)
- Online: 8-bit o 16-bit (latency priority)
- Mismo checkpoint, configuración dinámica

---

## 📊 Métricas de Éxito

### **Objetivos Técnicos**

Para considerar el roadmap exitoso:

1. **Quantización**: Encontrar método que preserve >95% accuracy con <50% VRAM
2. **Escalado**: Demostrar clara ventaja de 14B sobre 8B (o confirmar diminishing returns)
3. **Multi-Task**: Lograr modelo único que sea ≥90% de 3 modelos especializados
4. **Inference**: Lograr ≥2× speedup vs baseline actual

### **Objetivos de Negocio/Práctica**

1. **Democratización**: Permitir fine-tuning de calidad en hardware consumer (<$2000)
2. **Producción**: Proveer recetas claras para deployment
3. **ROI**: Demostrar cuándo vale la pena fine-tuning vs usar base model
4. **Generalización**: Principios que apliquen más allá de Qwen3

---

## 🤝 Recursos Necesarios

### **Hardware**

**Mínimo (continuar investigación actual):**
- 1× RTX 4090 (24GB) o similar
- Permite: hasta 8B-QLoRA, 4B-SFT

**Óptimo (FASE 4-5):**
- 2× RTX 4090 (48GB combinado)
- Permite: 14B-QLoRA, algunos experimentos 14B-SFT

**Ideal (roadmap completo):**
- 1× A100 40GB o H100
- Permite: 14B-SFT, 32B-QLoRA, experiments más grandes

### **Tiempo**

**Conservative estimate:**
- FASE 4: 2 meses (1 persona full-time)
- FASE 5: 2 meses (1 persona full-time)
- FASE 6-7: 3-4 meses (1 persona full-time)
- Total: ~8-10 meses para prioridades altas

**Con team de 2:**
- Reducir a ~5-6 meses paralelizando experimentos

### **Datasets**

**Nuevos datasets a adquirir:**
- HumanEval, MBPP (code) - gratuitos
- MT-Bench data - gratuito
- Domain-specific: depende del dominio elegido

---

## 🎓 Publicaciones Potenciales

### **Papers Planeados**

1. **"Advanced Quantization for Fine-Tuned LLMs: A Comprehensive Study"**
   - Based on FASE 4
   - Target: NeurIPS/ICML workshops
   - Timeline: Draft en 3 meses

2. **"Scaling Laws for Quantized Parameter-Efficient Fine-Tuning"**
   - Based on FASE 5
   - Target: ICLR/ACL
   - Timeline: Draft en 6 meses

3. **"Production-Ready LLM Deployment: From Training to Serving"**
   - Based on FASE 7
   - Target: MLSys, Systems for ML workshop
   - Timeline: Draft en 9 meses

### **Blog Posts / Technical Reports**

- Monthly progress updates en blog
- Detailed tutorials para cada método
- Comparison guides para practitioners

---

## ✅ Conclusión

### **Siguiente Acción Inmediata (Esta Semana)**

1. **Decidir**: ¿FASE 4 (Quantización) o FASE 5 (Escalado)?
   - Recomendación: **FASE 4** (mayor impacto inmediato)
   - Motivo: PTQ demostró ser problemática, necesitamos alternativa

2. **Setup**: Implementar QAT básica
   - Fork de `01_Train.py` → `01_Train_QAT.py`
   - Integrar Hugging Face QAT utilities
   - Target: 4B-QAT-OpenMath como proof-of-concept

3. **Baseline**: Ejecutar multi-seed validation
   - 3 seeds de 4B-SFT-OpenMath
   - Establecer error bars en resultados actuales
   - Publicar uncertainty ranges

### **Preguntas para Discutir**

1. ¿Prioridad en mejorar resultados actuales vs expandir a nuevos dominios?
2. ¿Hardware upgrade factible? (FASE 5 requiere más VRAM)
3. ¿Target de publicación? (determina timeline y profundidad)
4. ¿Interés en cross-architecture validation? (Llama vs Qwen)

### **Mensaje Final**

Tienen una base sólida con **3 fases completadas y 12 configuraciones evaluadas**. Los hallazgos son claros y accionables. El roadmap propuesto **prioriza resolver los problemas identificados** (quantización, escalado) antes de expandir a nuevos dominios.

**Recomendación estratégica**: 
- Próximos 2-3 meses: **FASE 4 (Quantización avanzada)**
- Meses 4-6: **FASE 5 (14B experiments)**
- Meses 7+: Expandir según resultados

Esto maximiza impact científico y valor práctico. 🚀

---

**¿Preguntas? ¿Ajustes al roadmap?**
