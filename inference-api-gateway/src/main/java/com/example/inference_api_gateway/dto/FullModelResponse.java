package com.example.inference_api_gateway.dto;

public record FullModelResponse(String predicted_category, Double confidence) {}
