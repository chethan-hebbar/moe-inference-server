package com.example.inference_api_gateway.dto;

import java.util.List;

public record GatingRequest(List<List<Double>> embeddings) {}
