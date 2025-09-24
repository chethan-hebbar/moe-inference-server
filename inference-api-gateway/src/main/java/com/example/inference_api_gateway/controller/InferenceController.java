package com.example.inference_api_gateway.controller;

import com.example.inference_api_gateway.dto.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.Collections;
import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/v1")
public class InferenceController {

    private final WebClient.Builder webClientBuilder;
    private final WebClient gatingServiceClient;
    private final WebClient embeddingServiceClient;
    private final WebClient classifierServiceClient;

    public InferenceController(WebClient.Builder webClientBuilder) {
        this.webClientBuilder = webClientBuilder;
        this.gatingServiceClient = webClientBuilder.baseUrl("http://gating-service:8000").codecs(clientCodecConfigurer -> clientCodecConfigurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024)).build();
        this.embeddingServiceClient = webClientBuilder.baseUrl("http://embedding-service:8000").codecs(clientCodecConfigurer -> clientCodecConfigurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024)).build();
        this.classifierServiceClient = webClientBuilder.baseUrl("http://classifier-service:8000").codecs(clientCodecConfigurer -> clientCodecConfigurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024)).build();
    }

    @PostMapping("/infer")
    public Mono<FinalResponse> performInference(@RequestBody InferenceRequest request) {

        return this.gatingServiceClient.post().uri("/tokenize")
                .bodyValue(new TokenizeRequest(request.text())).retrieve().bodyToMono(TokenizeResponse.class)

                .flatMap(tokenizeResponse ->
                        this.embeddingServiceClient.post().uri("/embed")
                                .bodyValue(new EmbeddingRequest(tokenizeResponse.token_indices())).retrieve().bodyToMono(EmbeddingResponse.class)
                )

                .flatMap(embeddingResponse -> {
                    Mono<GatingResponse> gatingResponseMono = this.gatingServiceClient.post().uri("/route")
                            .bodyValue(new GatingRequest(embeddingResponse.embeddings())).retrieve().bodyToMono(GatingResponse.class);
                    return Mono.zip(Mono.just(embeddingResponse), gatingResponseMono);
                })

                .flatMap(tuple -> {
                    EmbeddingResponse embeddingResponse = tuple.getT1();
                    GatingResponse gatingResponse = tuple.getT2();
                    ExpertRequest expertRequest = new ExpertRequest(embeddingResponse.embeddings());

                    return Flux.fromIterable(gatingResponse.top_k_indices())
                            .flatMap(expertId -> callExpertService(expertId, expertRequest))
                            .collectList()
                            .map(expertResponses -> {
                                double[][] combined = combineExpertOutputs(expertResponses, gatingResponse.top_k_weights());
                                return poolOutputs(combined);
                            });
                })

                // --- FINAL STEP: Call the Classifier Service ---
                .flatMap(pooledVector -> {
                    ClassifierRequest classifierRequest = new ClassifierRequest(pooledVector);
                    return this.classifierServiceClient.post().uri("/classify")
                            .bodyValue(classifierRequest).retrieve().bodyToMono(ClassifierResponse.class);
                })

                // --- Interpret the final logits ---
                .map(this::interpretFinalLogits);
    }

    private Mono<ExpertResponse> callExpertService(int expertId, ExpertRequest expertRequest) {
        String expertUrl = "http://expert-" + expertId + ":8001";
        WebClient expertClient = this.webClientBuilder.baseUrl(expertUrl).codecs(clientCodecConfigurer -> clientCodecConfigurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024)).build();
        return expertClient.post().uri("/process").bodyValue(expertRequest).retrieve().bodyToMono(ExpertResponse.class);
    }

    private double[][] combineExpertOutputs(List<ExpertResponse> responses, List<Double> weights) {
        if (responses.isEmpty()) return new double[0][0];
        int numTokens = responses.get(0).processed_representations().size();
        int dModel = responses.get(0).processed_representations().get(0).size();
        double[][] combined = new double[numTokens][dModel];

        for (int i = 0; i < responses.size(); i++) {
            double weight = weights.get(i);
            List<List<Double>> expertOutput = responses.get(i).processed_representations();
            for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
                for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {
                    combined[tokenIdx][dimIdx] += expertOutput.get(tokenIdx).get(dimIdx) * weight;
                }
            }
        }
        return combined;
    }

    private List<Double> poolOutputs(double[][] combinedRepresentations) {
        if (combinedRepresentations.length == 0) return Collections.emptyList();
        int numTokens = combinedRepresentations.length;
        int dModel = combinedRepresentations[0].length;
        Double[] pooled = new Double[dModel];
        for(int i=0; i<dModel; i++) pooled[i] = 0.0;

        for (int tokenIdx = 0; tokenIdx < numTokens; tokenIdx++) {
            for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {
                pooled[dimIdx] += combinedRepresentations[tokenIdx][dimIdx];
            }
        }
        for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {
            pooled[dimIdx] /= numTokens;
        }
        return List.of(pooled);
    }

    private FinalResponse interpretFinalLogits(ClassifierResponse classifierResponse) {
        List<Double> logits = classifierResponse.logits();
        // Apply Softmax to convert logits to probabilities
        double[] probabilities = new double[logits.size()];
        double maxLogit = Collections.max(logits);
        double sumExp = 0.0;
        for(int i=0; i<logits.size(); i++) {
            probabilities[i] = Math.exp(logits.get(i) - maxLogit); // Subtract max for numerical stability
            sumExp += probabilities[i];
        }
        for(int i=0; i<logits.size(); i++) {
            probabilities[i] /= sumExp;
        }

        // Find the index of the highest probability
        int predictedClassIndex = 0;
        double maxProb = -1.0;
        for(int i=0; i<probabilities.length; i++) {
            if(probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                predictedClassIndex = i;
            }
        }

        String[] categories = {"World", "Sports", "Business", "Sci/Tech"};
        String predictedCategory = categories[predictedClassIndex];

        return new FinalResponse(predictedCategory, maxProb);
    }
}