# Neural Network Visualization Backend

A FastAPI-based backend service that provides visualization capabilities for neural network layers and their activations. This service is designed to help understand and analyze how neural networks process and interpret data. This is the backend component of the CNN Visualizer project.

> **Note**: This is the backend repository. The frontend repository can be found at [CNN Visualizer Frontend](https://github.com/codeflamer/cnn-visualizer)

## Features

- RESTful API endpoints for accessing neural network layer information
- Layer-wise visualization capabilities
- CORS support for frontend integration
- Docker support for easy deployment
- Redis caching for improved performance

## Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Redis (optional, for caching)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/codeflamer/cnn-visualizer-backend
cd cnn-visualizer-backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

1. Start the server:

```bash
python main.py
```

The server will start at `http://localhost:8000`

### API Endpoints

- `GET /`: Welcome message
- `GET /all_layers`: Retrieve information about all neural network layers
- `GET /layer/{idx}`: Get specific layer tensor information by index

### Docker Deployment

1. Build the Docker image:

```bash
docker build -t neural-network-viz-backend .
```

2. Run the container:

```bash
docker run -p 8000:8000 neural-network-viz-backend
```

## Project Structure

```
.
├── main.py              # FastAPI application entry point
├── util.py             # Utility functions and helper methods
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── .dockerignore     # Docker ignore rules
├── Rediscache/       # Redis caching implementation
└── modeldatapickle/  # Neural network model data
```

## Full Stack Architecture

This backend service is designed to work in conjunction with the [CNN Visualizer Frontend](https://github.com/codeflamer/cnn-visualizer), which provides a modern web interface for visualizing neural network layers and activations. The frontend is built with Next.js and TypeScript, offering an intuitive user experience for exploring CNN architectures.

## Dependencies

- FastAPI (0.104.1): Web framework for building APIs
- Uvicorn (0.24.0): ASGI server
- PyTorch (2.1.0): Deep learning framework
- NumPy (1.24.3): Numerical computing
- Matplotlib (3.7.1): Data visualization
- Python-multipart (0.0.6): File upload handling

## Development

### Environment Setup

1. Create a `.env` file in the root directory with necessary environment variables
2. Install development dependencies
3. Configure your IDE settings

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document functions and classes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
