package com.example.inference_api_gateway.dto;

import java.util.List;

public record FinalResponse(String predicted_category, Double confidence) {}