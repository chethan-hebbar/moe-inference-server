package com.example.inference_api_gateway.dto;

import java.util.List;

public record ClassifierRequest(List<Double> pooled_vector) {}
