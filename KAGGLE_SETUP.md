# FRACTAL on Kaggle: Setup Instructions

## Quick Start (2xT4 GPUs)

### 1. Create a new Kaggle Notebook

1. Go to https://kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2**
4. Settings → Internet → **ON**

### 2. Clone FRACTAL Repository

```bash
!git clone https://github.com/YOUR_USERNAME/FRACTAL.git
%cd FRACTAL
```

### 3. Install Dependencies

```bash
!pip install -q torch torchvision
!pip install -q fair-esm requests tqdm pyyaml typer gemmi
!pip install -q huggingface-hub  # For pushing to HF
!pip install -e .
```

### 4. Run Full Pipeline

```python
# Option 1: Full pipeline (1000 samples)
!python kaggle_pipeline.py --n-samples 1000 --hf-repo YOUR_USERNAME/fractal-3b

# Option 2: Quick test (100 samples, no HF push)
!python kaggle_pipeline.py --n-samples 100

# Option 3: Resume from checkpoint
!python kaggle_pipeline.py --skip-download --skip-preprocess --hf-repo YOUR_USERNAME/fractal-3b
```

### 5. Expected Runtime (2xT4)

- **Download 1000 structures**: ~10 minutes
- **Preprocessing**: ~15 minutes (with ESM tokenization)
- **Training (20 epochs)**: ~2-3 hours
- **Inference (3 examples)**: ~5 minutes
- **Total**: ~3-4 hours

### 6. Memory Management Tips

If you hit OOM:
```yaml
# Edit configs/train.yaml
data:
  batch_size: 1  # Already set
  
optim:
  grad_accum_steps: 32  # Increase from 16
  amp: true  # Keep enabled
```

### 7. Monitor Training

```python
# In a separate cell, run while training:
!watch -n 10 cat models/trained/best/metadata.json
```

### 8. View Results

```python
from IPython.display import HTML, display

# View interactive 3D structure
with open('example_1.html') as f:
    display(HTML(f.read()))
```

### 9. Download Results

```python
from google.colab import files

# Download trained model
!zip -r fractal_model.zip models/trained/best
files.download('fractal_model.zip')

# Download examples
for i in range(1, 4):
    files.download(f'example_{i}.pdb')
    files.download(f'example_{i}.html')
```

## HuggingFace Setup

Before running with `--hf-repo`:

```python
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token
```

## Troubleshooting

**OOM during training:**
- Reduce `batch_size` to 1
- Increase `grad_accum_steps` to 32 or 64
- Use smaller model: `esm2_t33_650M_UR50D`

**Download failures:**
- Increase `--timeout` in download script
- Retry failed downloads manually

**NaN loss:**
- Gradient clipping is enabled (1.0)
- Lower learning rate if needed

## Advanced: Multi-GPU Training

The code automatically uses both GPUs if available (DataParallel).
No changes needed!
