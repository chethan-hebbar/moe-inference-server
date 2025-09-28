# Project: Distributed Mixture-of-Experts (MoE) Inference Server

This project is a deep dive into building and serving a state-of-the-art AI model. It covers the end-to-end lifecycle, from implementing a Mixture of Experts (MoE) Transformer model from scratch in PyTorch to deploying it in a high-performance, distributed microservice architecture using Java, Python, and Docker.

## Core Architectural Workflows

This server exposes two API versions, each corresponding to a different backend architecture for performing inference.

### Workflow 1: The Fully Distributed MoE (`/api/v1/infer`)

This workflow demonstrates a classic microservice approach to model parallelism. The AI model is decomposed into 12 discrete, containerized services that work in concert:

*   **1 API Gateway (Java)**
*   **1 Tokenizer Service**
*   **1 Embedding Service**
*   **1 Gating Service**
*   **8 parallel Expert Services**
*   **1 Classifier Service**

The gateway orchestrates a complex, asynchronous fan-out/gather-back pattern to execute the inference. While this architecture is a powerful demonstration of distributed systems, **it omits the self-attention step** of the Transformer for simplicity, which impacts model accuracy but clearly showcases the MoE routing pattern.

### Workflow 2: The High-Fidelity "Smart Monolith" (`/api/v2/infer`)

This workflow prioritizes maximum model accuracy. It uses a simplified 2-service architecture:

*   **1 API Gateway (Java)**
*   **1 Full Model Service (Python)**

In this setup, a single, powerful AI service loads the entire trained model (`full_model.pth`) and executes the complete, uncompromised `forward` pass, including all self-attention and MoE layers. The API Gateway simply acts as a proxy, forwarding the request and returning the high-fidelity prediction.

## System Architecture

The overall system is managed by Docker Compose, with the API Gateway acting as the central orchestrator and single point of entry for both workflows.

## Technical Deep Dive & Core Competencies

### Phase 1: MoE Model Implementation (PyTorch)

*   **Custom MoE Layer:** Implemented a sparse `MoELayer` with top-k gating and a load balancing loss function from scratch, using advanced tensor manipulations for efficient sparse dispatch.
*   **Optimized Model Training:** Trained a 6-layer, 512-dimension MoE Transformer on a premium A100 GPU, analyzing training logs to identify the optimal model checkpoint and achieving **~92.2% validation accuracy** on the AG News dataset.
*   **Component Serialization:** Wrote logic to serialize the trained weights of each individual model component (`embedding`, `gating`, `experts`, `classifier`) for microservice deployment.

### Phase 2: Distributed Inference Server (Java, Python, Docker)

*   **Asynchronous Orchestration:** Utilized Project Reactor (`Mono`, `Flux`) in the Spring Boot Gateway to manage complex, multi-step asynchronous workflows, including the parallel fan-out to expert services.
*   **Infrastructure-as-Code (DRY):** The entire 13-service application is defined and managed via a single `docker-compose.all.yml` file. The infrastructure code was refactored to be fully DRY (Don't Repeat Yourself) by using a single universal `Dockerfile` and `requirements.txt` for all Python services, configured with build arguments.
*   **Advanced Debugging:** Successfully diagnosed and solved a wide array of professional-grade, real-world problems, including deep dependency conflicts ("dependency hell"), hardware architecture (`aarch64`) incompatibilities, Docker build context issues, and subtle Python import errors within containers.

## How to Run

### Prerequisites

1.  **Docker & Docker Compose:** Must be installed and running on your local machine.
2.  **Git:** Required to clone the repository.
3.  **Model Weights:** The trained model weights are required to run the services.

### Setup

1.  **Download Model Weights:** Download the compressed model weights from the link below.
    *   **[Download Weights Here](https://drive.google.com/file/d/12j5DQe26d4zOgaQ8isFAh_rQQmYqG5a3/view?usp=sharing)**
2.  **Place Weights:** Unzip the downloaded file. You will find a set of `.pth` files. Place these files into the appropriate service `app` directories. For example:
    *   Place `full_model.pth` inside `full-model-service/app/`.
    *   Place `gating_network.pth` inside `gating-service/app/`.
    *   Place all `expert_*.pth` files inside `expert-service/app/`.
    *   ...and so on for the `embedding` and `classifier` weights.
3.  **Clone the Repository:**
    ```bash
    git clone https://github.com/chethan-hebbar/moe-inference-server.git
    cd moe-inference-server
    ```

### Launching the System

The entire 13-service application can be launched with a single command from the **root directory of this repository.**

```bash
# Build and launch all services
docker-compose -f docker-compose.yml up --build