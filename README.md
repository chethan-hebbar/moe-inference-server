# Project: Distributed Mixture-of-Experts (MoE) Inference Server

This project is a deep dive into building and serving a state-of-the-art AI model. It covers the end-to-end lifecycle, from implementing a Mixture of Experts (MoE) Transformer model from scratch in PyTorch to deploying it in a high-performance, distributed microservice architecture using Java, Python, and Docker.

The primary goal is to demonstrate a comprehensive understanding of both frontier AI architectures and robust, scalable systems engineering principles.

## Key Features

*   **Custom MoE Model:** A from-scratch implementation of a Transformer model that replaces the standard Feed-Forward Network with a sparsely-gated Mixture of Experts layer.
*   **Distributed Microservice Architecture:** The trained model is decomposed and served by a network of 12 independent, containerized microservices.
*   **High-Performance Gateway:** A non-blocking, asynchronous API Gateway built in Java with Spring Boot WebFlux orchestrates the complex inference workflow.
*   **Parallel Expert Processing:** The system dynamically routes requests to a subset of expert services and processes them in parallel, simulating a real-world model parallelism strategy.
*   **Containerized & Composable:** The entire 12-service application is defined and managed by Docker Compose, allowing for a one-command launch (`docker-compose up`).

## System Architecture

The inference server is designed as a distributed system with a single point of entry. The API Gateway is the "brain" that orchestrates the calls to the various "expert" AI services.

![System Architecture Diagram](architecture.png)
*(Note: You should replace this with a real screenshot of your architectural diagram and name the file `architecture.png`)*

### Service Breakdown:

1.  **API Gateway (Java / Spring Boot WebFlux):**
    *   The public-facing entry point (`POST /api/v1/infer`).
    *   Orchestrates the entire inference pipeline using a reactive, non-blocking chain of asynchronous calls.
    *   Responsible for the final combination of expert outputs and classification.

2.  **Tokenizer Service (Python / FastAPI):**
    *   An endpoint (hosted on the Gating Service) that converts raw text into numerical token indices based on the trained vocabulary.

3.  **Embedding Service (Python / FastAPI):**
    *   Loads the trained `nn.Embedding` weights and `PositionalEncoding` logic.
    *   Converts token indices into high-dimensional vector representations.

4.  **Gating Service (Python / FastAPI):**
    *   Loads the trained MoE gating network weights.
    *   Receives embedding vectors and returns a routing decision: the `top_k` expert indices and their corresponding softmax scores (weights).

5.  **Expert Services (Python / FastAPI):**
    *   Eight identical, containerized services, each loading a different set of trained expert weights (`expert_0.pth` through `expert_7.pth`).
    *   Configured at runtime via an `EXPERT_ID` environment variable.
    *   Performs the core computational task on the embedding vectors.

6.  **Classifier Service (Python / FastAPI):**
    *   Loads the trained classification head (a single linear layer).
    *   Takes the final, pooled vector from the gateway and produces the final output logits for the classes.

## Technical Deep Dive & Core Competencies

### Phase 1: MoE Model Implementation (PyTorch)

*   **Custom MoE Layer:** Implemented a sparse `MoELayer` with top-k gating and a load balancing loss function from scratch. The implementation uses advanced PyTorch tensor manipulations (masks, `scatter_`, `index_add_`) for efficient, sparse dispatch and combination.
*   **Full Transformer Training:** Integrated the `MoELayer` into a full Transformer classification model and successfully trained it on the AG News dataset to >90% accuracy, proving the validity of the architecture.
*   **Component Serialization:** Wrote logic to serialize the trained weights of each individual model component (`embedding`, `gating`, `experts`, `classifier`) in preparation for microservice deployment.

### Phase 2: Distributed Inference Server (Java, Python, Docker)

*   **Asynchronous Orchestration:** Utilized Project Reactor (`Mono`, `Flux`) in the Spring Boot Gateway to manage a complex, multi-step asynchronous workflow, including a parallel "fan-out" and "gather-back" of requests to the expert services.
*   **Containerization & Service Discovery:** Each of the 12 services is containerized using a multi-stage Dockerfile for optimization. Services communicate over a shared Docker network, using service names for discovery, which is handled by Docker's internal DNS.
*   **Configuration Management:** Employed professional microservice patterns, such as using environment variables (`EXPERT_ID`) and `docker-compose.yml` to configure and launch multiple instances of a single, generic expert service image.
*   **Architectural Trade-offs:**
    *   A shared `common` library was created for all Python model definitions to maintain a single source of truth and avoid code duplication.
    *   **Note:** For the purpose of focusing on the distributed MoE FFN architecture, the self-attention component of the Transformer Encoder layers is currently omitted during inference. A future enhancement would be to implement a dedicated "Attention Service."

## How to Run

1.  **Prerequisites:** Docker and Docker Compose must be installed.
2.  **Model Weights:** This project requires the trained model weights to run. Download them from [LINK-TO-YOUR-GOOGLE-DRIVE-OR-S3-BUCKET] and place the `.pth` files into the appropriate service `app` directories (`embedding-service/app/`, `gating-service/app/`, etc.).
3.  **Clone the repository:** `git clone <your-repo-url>`
4.  **Navigate to the root directory:** `cd phase2_inference_server`
5.  **Launch the system:**
    ```bash
    docker-compose up --build
    ```
6.  **Wait** for all 12 services to build and start. Watch the logs until they are stable.
7.  **Send a request:** Open a new terminal and use `curl` to test the endpoint.
    ```bash
    curl -X POST http://localhost:8080/api/v1/infer \
    -H "Content-Type: application/json" \
    -d '{"text": "Your test sentence about sports, science, or business goes here."}'
    ```
