# Kaggle Skills Project

A lightweight scaffold for experimenting, training, and submitting models to Kaggle competitions with reproducible Python environments.

## Quick start

```bash
# 1. Create and activate a local venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install the locked dependencies
pip install -r requirements.lock

# 3. Train your model
make train          # or python src/train.py

# 4. Create a submission
make submit
```

## Project Structure

```
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── submit.py          # Submission script
│   ├── data/              # Data processing modules
│   └── models/            # Model definitions
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── outputs/               # Model outputs and submissions
├── requirements.lock      # Locked dependencies
├── requirements.txt       # Dependency specifications
├── Makefile              # Build automation
└── README.md             # This file
```

## Available Commands

- `make setup` - Create virtual environment and install dependencies
- `make install` - Install dependencies
- `make train` - Train the model
- `make submit` - Create submission file
- `make clean` - Clean up generated files

## Development

1. **Setup**: Run `make setup` to create the environment
2. **Implement**: Fill in the TODO sections in `src/train.py` and `src/submit.py`
3. **Test**: Add tests in the `tests/` directory
4. **Experiment**: Use notebooks in `notebooks/` for exploration

## Dependencies

This project uses the following key dependencies:
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **lightgbm**: Gradient boosting framework
- **joblib**: Model persistence

## License

MIT License - see [LICENSE](LICENSE) file for details.