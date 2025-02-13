# 🚀 Model Evaluation with SGLang  

This repository contains tools for setting up **SGLang** with **CUDA support** and evaluating various **language models** using a standardized benchmarking approach.  

## 📂 Repository Structure  
```
.
├── 📜 README.md  
├── 📓 Sglang_instruct_eval.ipynb    # Main benchmarking notebook  
├── 📁 Sample Generation/            # Contains model evaluation scores  
└── 📄 instruct Benchmark.txt        # Detailed benchmark outputs  
```

## ✅ Prerequisites  
- 🎮 **NVIDIA GPU** with CUDA support  
- 🐍 **Python 3.10** or later  
- 🖥️ **Linux** operating system  
- 💾 **Sufficient disk space** (~5GB for CUDA, ~2GB for PyTorch, ~1GB for SGLang)  

## ⚙️ Setup Instructions  

### 🔹 1. CUDA Installation  
```bash
# 🚀 Download CUDA 12.1  
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# 🛠️ Install CUDA (without driver)  
sudo sh cuda_12.1.0_530.30.02_linux.run

# 🔗 Add CUDA to PATH and LD_LIBRARY_PATH  
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 🔄 Make the PATH changes permanent  
echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### 🔹 2. PyTorch Installation  
```bash
# ⚡ Install PyTorch with CUDA 12.1 support  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 🔹 3. SGLang Installation  
```bash
# 🛠️ Install SGLang with all dependencies  
pip install "sglang[all]"

# 🚀 Install FlashInfer (optional, for better performance)  
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
```

## 📊 Model Benchmarking  

### 📌 Overview  
The repository includes a **comprehensive benchmarking system** that evaluates language models across **20 diverse questions**. The evaluation process includes:  
✅ **Automated model response generation**  
✅ **Response quality assessment**  
✅ **Score aggregation and analysis**  

### ▶️ Running the Benchmark  
1️⃣ Open `Sglang_instruct_eval.ipynb` in **Jupyter Notebook/Lab**  
2️⃣ Follow the notebook cells to:  
   - 🔄 Load and initialize models  
   - ✍️ Generate responses to benchmark questions  
   - 💾 Save results in **JSON format**  

### 🏆 Evaluation Methods  
The framework supports **two evaluation approaches**:  

1️⃣ **Automated Evaluation** 🤖  
   - Uses **larger models (ChatGPT/Claude) as evaluators**  
   - Systematic scoring based on predefined criteria  

2️⃣ **Manual Evaluation** 👨‍💻  
   - Direct **human assessment** of responses  
   - Qualitative and quantitative scoring  

### 📂 Output Format  
Results are stored in **two locations**:  
- 📁 `Sample Generation/` → Contains individual model scores  
- 📄 `instruct Benchmark.txt` → Detailed benchmark outputs  

## ✅ Verification Steps  
```bash
# 🎮 Verify CUDA installation  
nvcc --version  

# 🔍 Verify PyTorch CUDA support  
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 🛠️ Test SGLang  
python3 -c "import sglang as sgl"
```

## 🚀 Quick Start Example  
```python
import sglang as sgl
import asyncio

# 🧠 Initialize the engine with a model  
llm = sgl.Engine(model_path="NousResearch/Hermes-3-Llama-3.2-3B")
```

## ⚠️ Important Notes  
⚡ Make sure you have the appropriate **NVIDIA drivers installed** before CUDA installation  
⚡ The CUDA installer will **warn about driver installation** – you can ignore this if you already have compatible drivers  
⚡ If you encounter **permission issues**, you might need to use `sudo` for some commands  
⚡ For production use, consider using a **virtual environment**  
⚡ Benchmark results may **vary** based on model versions and evaluation criteria  

## 🛠️ Troubleshooting  
If you encounter any issues:  
🔹 Verify your **NVIDIA drivers** are properly installed  
🔹 Check if **CUDA paths** are correctly set in your environment  
🔹 Ensure your **Python version** is compatible  
🔹 Make sure you have **sufficient disk space**  
🔹 Check **JSON output format** if evaluation results aren’t being saved properly  

## 🤝 Contributing  
Feel free to contribute by:  
🔹 **Adding new benchmark questions**  
🔹 **Implementing additional evaluation metrics**  
🔹 **Testing with different models**  
🔹 **Improving documentation**  

## 📚 References  
🔗 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)  
🔗 [PyTorch Installation](https://pytorch.org/get-started/locally/)  
🔗 [SGLang Documentation](https://github.com/build-ai-applications/sglang)  
🔗 [FlashInfer](https://flashinfer.ai)  

## 📜 License  
📄 This setup guide and benchmarking framework is provided under the **Apache 2.0**.  
