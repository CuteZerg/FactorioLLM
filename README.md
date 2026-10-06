# FactorioLLM: Text-to-Blueprint Generator

![Factorio](https://img.shields.io/badge/Game-Factorio-orange)
![Machine Learning](https://img.shields.io/badge/AI-Generative_NLP-blue)
![Python](https://img.shields.io/badge/Language-Python_3.12-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**FactorioLLM** is an AI-powered tool that generates functional, ready-to-use blueprints for the game Factorio based on natural language prompts. 

Instead of manually placing entities or searching the web for specific layouts, you can just type: 
> *"Create an expandable copper smelting line for 24 furnaces using red belts,"* 

...and receive a working blueprint string instantly.

---

## How It Works (The Core Concept)

Factorio blueprints are essentially JSON objects containing raw coordinates, compressed via `zlib` and encoded in `base64`. 

**The Challenge:** Large Language Models (LLMs) lack spatial reasoning. Asking an LLM to output a raw JSON blueprint with exact x/y coordinates for hundreds of entities results in broken, non-functional layouts. Models do not inherently understand "tileability" or belt alignment.

**The Solution:** FactorioLLM uses a **Text-to-Code** approach. 
Instead of predicting JSON coordinates, the LLM generates a **Python script** utilizing the [factorio-draftsman](https://github.com/redruin1/factorio-draftsman) library. Concepts like spacing, alignment, and ratios are naturally handled through Python `for` loops, variables, and math. The backend then safely executes this script to compile a flawless blueprint string.

### The Agentic Loop Pipeline

The project features an automatic self-healing loop:

```mermaid
graph TD;
    A[User Prompt] --> B(LLM: Llama-3 8B Fine-Tuned);
    B -->|Generates Python Script| C{Draftsman API / Execution};
    C -->|Syntax / Logic Error| B;
    C -->|Successful Compilation| D[Draftsman Output];
    D -->|Generates Base64| E[Ready Blueprint String];
    E --> A;
```

---

## Tech Stack

* **AI Models:** Fine-tuned Code-specific LLMs (`unsloth/llama-3-8b-Instruct-bnb-4bit`).
* **Training:** `Unsloth` for 2x faster 4-bit LoRA training, Hugging Face `transformers`, `PEFT` (LoRA).
* **Dataset Generation:** Google Gemini (`google-genai`), `factorio-draftsman`.
* **Package Management:** `uv`

---

## Getting Started

*(Note: The project is currently in active development. These instructions represent the local testing setup).*

### Prerequisites
* Python 3.12+
* `uv` package manager
* Factorio (for testing the generated blueprints)
* A GPU with CUDA support for training.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FactorioLLM.git
   cd FactorioLLM
   ```

2. Sync dependencies using `uv`:
   ```bash
   uv sync
   ```

### Training

To train the model on the generated dataset:
```bash
uv run python src/factoriollm/train_model.py
```

---

## Project Roadmap

- [x] **Phase 0: Proof of Concept** - Manual validation of Draftsman code generation.
- [x] **Phase 1: Data Engineering** - Fetching blueprints from factorioprints and decompiling them.
- [x] **Phase 2: Synthetic Prompts** - Using Google Gemini to attach diverse human prompts to the decompiled scripts.
- [x] **Phase 3: Fine-Tuning** - Training a LoRA adapter for Llama-3 to understand Factorio logic and Draftsman syntax. *(IN PROGRESS)*
- [ ] **Phase 4: Inference & Agentic Loop** - Implementing the self-healing sandbox execution.
- [ ] **Phase 5: User Interface** - Releasing a Web UI or a local app.

---

## Contributing

Contributions are welcome! If you love Factorio and Machine Learning, feel free to open an issue or submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Disclaimer: This project is created for educational and community purposes. "Factorio" is a registered trademark of Wube Software Ltd.*