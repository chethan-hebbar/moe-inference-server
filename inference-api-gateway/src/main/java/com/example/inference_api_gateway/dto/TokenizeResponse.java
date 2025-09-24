package com.example.inference_api_gateway.dto;

import java.util.List;

public record TokenizeResponse(List<Integer> token_indices) {}
