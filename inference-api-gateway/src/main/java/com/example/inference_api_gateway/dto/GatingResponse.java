package com.example.inference_api_gateway.dto;

import java.util.List;

public record GatingResponse(List<Integer> top_k_indices, List<Double> top_k_weights) {}
