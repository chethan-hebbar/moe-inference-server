package com.example.inference_api_gateway.dto;

import java.util.List;

public record ExpertRequest(List<List<Double>> embeddings) {}
