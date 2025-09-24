package com.example.inference_api_gateway.dto;

import java.util.List;

public record ExpertResponse(List<List<Double>> processed_representations) {}
