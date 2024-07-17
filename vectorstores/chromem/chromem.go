package chromem

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	chromemgo "github.com/philippgille/chromem-go"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

// Store is a wrapper around chromem-go.
type Store struct {
	// DB options
	persistent  bool
	persistPath string
	compress    bool

	// Collection options
	defaultCollectionName string // langchaingo "namespace", can be overwritten in each document addition/query
	embedder              embeddings.Embedder

	// Created based on above options
	db *chromemgo.DB
	ef chromemgo.EmbeddingFunc
}

var _ vectorstores.VectorStore = (*Store)(nil)

// New creates a new [Store] object with the passed options. Only the embedder
// is required, all other options either have defaults or are optional. If you
// don't set the `WithDefaultNamespace` option then you have to provide the
// namespace option in each call to [Store.AddDocuments] and
// [Store.SimilaritySearch].
func New(opts ...Option) (*Store, error) {
	s, err := applyClientOptions(opts...)
	if err != nil {
		return s, err
	}

	if s.persistent {
		s.db, err = chromemgo.NewPersistentDB(s.persistPath, s.compress)
		if err != nil {
			return nil, fmt.Errorf("couldn't create persistent DB: %w", err)
		}
	} else {
		s.db = chromemgo.NewDB()
	}

	// For the embedding func we only convert the langchaingo *query* embedder,
	// because for document embedding we can use it as is.
	s.ef = func(ctx context.Context, text string) ([]float32, error) {
		return s.embedder.EmbedQuery(ctx, text)
	}

	return s, nil
}

// AddDocuments adds the documents to the chromem-go DB and returns the IDs of
// the added documents. More precisely, the documents are added to the
// collection that's associated to the namespace that's set as the store's
// configured default namespace or passed via options to this method. One of the
// two namespaces must be set. If both are set, the latter takes precedence.
func (s *Store) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) ([]string, error) {
	opts := getOptions(vectorstores.Options{NameSpace: s.defaultCollectionName}, options...)
	if err := validateOptions(opts); err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	// Get or create collection. As the options parameter allows to pass a new namespace
	// on each call, a new collection might have to be created. Existing collections
	// come from either prior document additions or from a DB with persistence.
	c, err := s.db.GetOrCreateCollection(opts.NameSpace, nil, s.ef)
	if err != nil {
		return nil, fmt.Errorf("couldn't get or create collection: %w", err)
	}

	// While we might be able to benefit from chromem-go's concurrency (letting
	// it create the embeddings in parallel) we don't know whether the
	// langchaingo embedder implementation that's injected has some batch
	// optimizations which might be more efficient. So we create the embeddings
	// in advance and then add documents one by one.

	texts := make([]string, 0, len(docs))
	for _, doc := range docs {
		texts = append(texts, doc.PageContent)
	}
	embeddings, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("couldn't embed documents: %w", err)
	}

	ids := make([]string, 0, len(docs))
	for i, doc := range docs {
		id := uuid.NewString()
		// So far chromem-go only supports string values in the metadata.
		// TODO: As a temporary workaround until other types are allowed in
		// chromem-go, we could convert from some (simple) types to string here.
		var metadata map[string]string
		if len(doc.Metadata) > 0 {
			metadata = map[string]string{}
			for k, v := range doc.Metadata {
				vString, ok := v.(string)
				if !ok {
					return nil, errors.New("only string values are supported in the metadata map")
				}
				metadata[k] = vString
			}
		}

		err = c.AddDocument(ctx, chromemgo.Document{
			ID:        id,
			Metadata:  metadata,
			Embedding: embeddings[i],
			Content:   doc.PageContent,
		})
		if err != nil {
			return nil, fmt.Errorf("couldn't add document: %w", err)
		}

		ids = append(ids, id)
	}

	return ids, nil
}

// SimilaritySearch searches for similar documents in the chromem-go DB using
// cosine similarity and returns them. More precisely, the documents are
// searched in the collection that's associated to the namespace that's set as
// the store's configured default namespace or passed via options to this
// method. One of the two namespaces must be set. If both are set, the latter
// takes precedence.
func (s *Store) SimilaritySearch(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
	opts := getOptions(vectorstores.Options{NameSpace: s.defaultCollectionName}, options...)
	if err := validateOptions(opts); err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}
	var where map[string]string
	if opts.Filters != nil {
		where = opts.Filters.(map[string]string)
	}

	// Get collection
	c := s.db.GetCollection(opts.NameSpace, s.ef)
	if c == nil {
		return nil, errors.New("namespace doesn't exist - create it by adding documents to it first")
	}

	// chromem-go doesn't support a threshold yet, so we fetch the desired
	// number of docs first, and filter by threshold later
	docs, err := c.Query(ctx, query, numDocuments, where, nil)
	if err != nil {
		return nil, fmt.Errorf("couldn't query collection: %w", err)
	}

	// Filter by threshold
	var res []schema.Document
	for _, doc := range docs {
		if doc.Similarity >= opts.ScoreThreshold {
			var metadata map[string]any
			if len(doc.Metadata) > 0 {
				metadata = map[string]any{}
				for k, v := range doc.Metadata {
					metadata[k] = v
				}
			}
			res = append(res, schema.Document{
				PageContent: doc.Content,
				Metadata:    metadata,
				Score:       doc.Similarity,
			})
		}
	}

	return res, nil
}

func getOptions(defaults vectorstores.Options, opts ...vectorstores.Option) vectorstores.Options {
	res := defaults
	for _, opt := range opts {
		opt(&res)
	}

	return res
}

// validateOptions currently doesn't differentiate between options for
// [Store.AddDocuments] and [Store.SimilaritySearch]. We could be more strict,
// but due to langchaingo not differentiating the options, users potentially
// create only one options object and expect it to work in both method calls.
func validateOptions(opts vectorstores.Options) error {
	// We don't support all options yet
	if opts.Deduplicater != nil || opts.Embedder != nil {
		return errors.New("unsupported options")
	}

	// Validate individual values
	if opts.NameSpace == "" {
		return errors.New("namespace is empty")
	}
	if opts.ScoreThreshold < 0 || opts.ScoreThreshold > 1 {
		return errors.New("score threshold must be between 0 and 1")
	}
	// chromem-go supports filters for metadata and document content. Most
	// vector store implementations in langchaingo seem to focus on metadata
	// filters so we focus on that first.
	// TODO: Implement a way to make *both* types of filters usable via the
	// filters option.
	if opts.Filters != nil {
		if _, ok := opts.Filters.(map[string]string); !ok {
			return errors.New("filters must be of type map[string]string")
		}
	}

	return nil
}

// TODO: Export and Import
