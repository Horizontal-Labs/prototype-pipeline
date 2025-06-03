# Argument Mining API

A FastAPI-based service for analyzing argumentative structures in text using state-of-the-art NLP models.

![Python Version](https://img.shields.io/badge/python-3.12.1-blue.svg)
![Test Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- Identification of argument components (claims, premises)
- Analysis of argumentative relations
- RESTful API with FastAPI
- Powered by BERT and spaCy models
- Comprehensive test suite

## Requirements

- Python 3.12.1 or higher
- Hugging Face account and API token
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Horizontal-Labs/pipeline
cd pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_lg
```

5. Set up environment variables:
Create a `.env` file in the project root with:
```
HF_TOKEN=your_huggingface_token_here
```

## Usage

1. Start the server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example API Request

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "Global warming is a serious threat. Since temperatures are rising worldwide, we need to act now."
    }
)

print(response.json())
```

## API Endpoints

### POST /analyze
Analyzes text for argument components and relations.

Request body:
```json
{
    "text": "string"
}
```

Response:
```json
{
    "components": [
        {
            "text": "string",
            "type": "claim|premise|non-argument",
            "confidence": 0.95
        }
    ],
    "relations": [
        {
            "source_idx": 0,
            "target_idx": 1,
            "relation_type": "support|attack",
            "confidence": 0.8
        }
    ]
}
```

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy"
}
```

## Project Structure

```
ArgumentMining/
├── app/
│   ├── __init__.py
│   ├── models.py      # Pydantic models
│   ├── routes.py      # API endpoints
│   └── services.py    # Core business logic
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_argument_mining.py
│   └── test_model.py
├── main.py            # Application entry point
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:
```bash
pytest
```

For test coverage report:
```bash
pytest --cov=. --cov-report=term-missing
```

Current test coverage: 93%

## Development

- The project uses FastAPI for the web framework
- Models are loaded using the Hugging Face Transformers library
- Argument component classification uses BERT
- Sentence segmentation uses spaCy
- Testing uses pytest with pytest-cov for coverage reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/)
- [BERT](https://github.com/google-research/bert)