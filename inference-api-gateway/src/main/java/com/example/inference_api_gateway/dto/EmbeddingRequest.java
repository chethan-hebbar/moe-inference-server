package com.example.inference_api_gateway.dto;

import java.util.List;

public record EmbeddingRequest(List<Integer> token_indices) {}
