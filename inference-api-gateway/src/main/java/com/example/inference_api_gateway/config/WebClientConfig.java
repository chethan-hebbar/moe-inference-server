package com.example.inference_api_gateway.config;


import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient.Builder webClientBuilder() {
        // By creating a WebClient.Builder bean, Spring Boot will auto-configure
        // it with sensible defaults. We can then inject this builder wherever we need it.
        return WebClient.builder();
    }
}