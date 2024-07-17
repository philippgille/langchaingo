package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chromem"
)

func main() {
	ctx := context.Background()

	llm, err := openai.New(openai.WithEmbeddingModel("text-embedding-3-small"))
	if err != nil {
		log.Fatalf("Couldn't create OpenAI client: %v\n", err)
	}
	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatalf("Couldn't create embedder: %v\n", err)
	}

	// Create a new chromem-go vector store.
	store, err := chromem.New(
		chromem.WithEmbedder(embedder),
		// The default namespace is optional, but setting this option allows to
		// omit the namespace in the `AddDocuments` and `SimilaritySearch` calls
		chromem.WithDefaultNamespace("langchaingo"),
	)
	if err != nil {
		log.Fatalf("Couldn't create chromem-go store: %v\n", err)
	}

	type meta = map[string]any

	// Add documents to the vector store. So far chromem-go only supports string
	// values metadata maps.
	docs := []schema.Document{
		{PageContent: "Tokyo", Metadata: meta{"population": "9.7", "area": "622"}},
		{PageContent: "Kyoto", Metadata: meta{"population": "1.46", "area": "828"}},
		{PageContent: "Hiroshima", Metadata: meta{"population": "1.2", "area": "905"}},
		{PageContent: "Kazuno", Metadata: meta{"population": "0.04", "area": "707"}},
		{PageContent: "Nagoya", Metadata: meta{"population": "2.3", "area": "326"}},
		{PageContent: "Toyota", Metadata: meta{"population": "0.42", "area": "918"}},
		{PageContent: "Fukuoka", Metadata: meta{"population": "1.59", "area": "341"}},
		{PageContent: "Paris", Metadata: meta{"population": "11", "area": "105"}},
		{PageContent: "London", Metadata: meta{"population": "9.5", "area": "1572"}},
		{PageContent: "Santiago", Metadata: meta{"population": "6.9", "area": "641"}},
		{PageContent: "Buenos Aires", Metadata: meta{"population": "15.5", "area": "203"}},
		{PageContent: "Rio de Janeiro", Metadata: meta{"population": "13.7", "area": "1200"}},
		{PageContent: "Sao Paulo", Metadata: meta{"population": "22.6", "area": "1523"}},
	}
	_, err = store.AddDocuments(context.Background(), docs)
	if err != nil {
		log.Fatalf("Couldn't add documents: %v\n", err)
	}

	type exampleCase struct {
		name         string
		query        string
		numDocuments int
		options      []vectorstores.Option
	}

	exampleCases := []exampleCase{
		{
			name:         "Up to 5 Cities in Japan",
			query:        "Which of these cities are located in Japan?",
			numDocuments: 5,
			// options: []vectorstores.Option{
			// 	vectorstores.WithScoreThreshold(0.8),
			// },
		},
		{
			name:         "A City in South America",
			query:        "Which of these cities are located in South America?",
			numDocuments: 1,
			// options: []vectorstores.Option{
			// 	vectorstores.WithScoreThreshold(0.8),
			// },
		},
		{
			name:         "A City in South America with an area of 1523 square km",
			query:        "Which city is located in South America?",
			numDocuments: len(docs), // The filter already limits the result
			options: []vectorstores.Option{
				vectorstores.WithFilters(map[string]string{"area": "1523"}), // Sao Paolo
			},
		},
	}

	// Run the example cases
	for i, ec := range exampleCases {
		docs, err := store.SimilaritySearch(ctx, ec.query, ec.numDocuments, ec.options...)
		if err != nil {
			log.Fatalf("Couldn't find similar documents: %v\n", err)
		}

		// Print result after each example case
		texts := make([]string, 0, len(docs))
		for _, doc := range docs {
			texts = append(texts, doc.PageContent+fmt.Sprintf(" (similarity %f)", doc.Score))
		}
		fmt.Printf("%d. case: %s\n", i+1, ec.name)
		fmt.Printf("   result:\n   - %s\n", strings.Join(texts, "\n   - "))
	}
}
