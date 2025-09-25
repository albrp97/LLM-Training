@Lobato and I have been researching **how to make local models truly feasible**—and answering a key question when building LLM solutions: **do we need big API models, or can a small, fine-tuned local model deliver better value?** 🤔💸🚀

### TL;DR

* **API ≠ automatic best choice.** With a **locally fine-tuned 0.6B**, we reached **quality comparable to a larger 1.7B base** while cutting **latency by up to ~99%**, avoiding API price swings and rate limits. **More control, lower opex, predictable governance.** 🔒⚡📉
* **Task fit matters.** **Context-anchored tasks** (RAG/extractive QA) gain the most from SFT; **math/logic-heavy tasks** benefit less from size-preserving SFT.
* **Cross-task effects:** full FT can yield strong **positive transfer** or **catastrophic forgetting**—plan training and evaluation with that in mind. 🧪
* **Full fine-tuning > LoRA** when VRAM allows—consistently **better efficacy** and **often faster** in our runs. 🏎️📈
* **LoRA rank:** bigger rank **didn’t consistently help**. If you must use LoRA, **~256** worked well in practice. 🧩

### What we tested 🧪

* **Models:** Qwen3-0.6B (base, LoRA r∈{32…1024}, full FT) and Qwen3-1.7B (base). 🤖
* **Tasks:** ARC (MCQ), OpenMathInstruct-2 (numeric), SQuAD v2 (extractive QA). 🧠📚
* **Procedure:** Supervised FT (TRL/Transformers), deterministic decoding, reporting **quality + wall-clock latency** per sample. 🧬⏱️

### Highlights ✨

* **Local 0.6B + full FT** vs 0.6B base: 🏠⚡

  * **Latency:** ↓ up to **~99.2%** (e.g., numeric tasks) and **~94.7%** on MCQ. ⏱️⬇️
  * **Quality:** **SQuAD v2 F1** ↑ **~177.6%** when tuned on SQuAD; **ARC macro-F1** ↑ **~4.8%** when tuned on OpenMath (transfer effect). 📈🏆
* **Full FT vs best LoRA:** 🆚

  * On numeric tasks, full FT cut error by **~7,115 absolute units** vs best LoRA **and** ran **~1.45s faster** per sample. 🔢⚡
  * On extractive QA, full FT beat best LoRA by **+18.36 F1**, with a tiny latency trade-off (~0.03s). 📝👌
* **LoRA rank:** Correlations with score were ~0 or inconsistent; **use r≈256** by default if constrained. 🎚️✅
* **Transfer & forgetting:** 🔁🧠

  * **Positive transfer:** ARC-tuned → SQuAD saw **large F1 gains** in our setting. ➕🚀
  * **Forgetting:** OpenMath-tuned → SQuAD showed **~26.5% F1 drop** (single-task FT can erode other skills). ⚠️📉

### Practical takeaways 💡

* **If you rely on APIs today:** evaluate a **small, locally fine-tuned model** first—you’ll likely **save cost**, **cut latency**, and **gain control** over versions/compliance. 💸 ⚡
* **When VRAM is available:** prefer **full FT** over adapters for best results. 🖥️ ✅
* **When VRAM is tight or you need multiple domains:** use **LoRA** (hot-swappable adapters), and start at **r≈256**. 🔁 🎚️
* **Match FT to task type:** 🎯

  * **Best with FT:** context-anchored/RAG/extractive QA (formatting, span selection, instruction alignment). 📚 🔗
  * **Less uplift from size-preserving FT:** math/logic-heavy workloads—consider scaling base size or complementary techniques. 🧮 ⬆️

We’ll keep publishing new experiments on **fine-tuning strategies**, **quantization**, and **inference optimizations** to push smaller models further. 🔬

🔗 **Repo:** <add your repo link>
📔 **Complete journal (PDF):** <add your journal link>

—
#LLM #FineTuning #LoRA #PEFT #Latency #CostOptimization #RAG #MLOps #OpenSource #Qwen #NLP #AIEngineering #AI #MachineLearning #OnDeviceAI #EdgeAI #PrivateAI #Quantization #ModelCompression #Distillation #Inference #ModelServing #LatencyReduction #CostSavings #Governance #LLMEval #Benchmarking
