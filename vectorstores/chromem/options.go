package chromem

import (
	"errors"

	"github.com/tmc/langchaingo/embeddings"
)

var (
	defaultPersistPath = "./langchaingo"
	defaultCompress    = false
)

// Option is a function type that can be used to modify the client.
type Option func(p *Store)

// WithPersistence enables immediate document-level persistence. With this
// persistence there will be one file written for each added document.
// The files will be stored in the given path. If empty, "./langchaingo" will be
// used.
// If compress is true, the files will be compressed using gzip.
//
// If documents have been persisted during a previous execution, then creating a
// new store will initialize it by loading all documents from these files.
//
// As an alternative to this persistence option you can also export the DB to a
// single file using the [Store.Export] method, which also works when not using
// this persistence option.
func WithPersistence(path string, compress bool) Option {
	return func(s *Store) {
		s.persistent = true
		s.persistPath = path
		s.compress = compress
	}
}

// WithDefaultNamespace sets the default namespace for the chromem-go collection.
// If you don't set this, then you have to provide the namespace option for all
// [Store.AddDocuments] and [Store.SimilaritySearch] calls.
// Each namespace maps to a separate collection in chromem-go.
func WithDefaultNamespace(defaultNamespace string) Option {
	return func(s *Store) {
		s.defaultCollectionName = defaultNamespace
	}
}

// WithEmbedder sets the embedder to use for document and query embedding.
// It's a required option and creating a chromem store without this option will
// return an error.
func WithEmbedder(embedder embeddings.Embedder) Option {
	return func(p *Store) {
		p.embedder = embedder
	}
}

func applyClientOptions(opts ...Option) (*Store, error) {
	// Initialize with defaults
	s := &Store{
		persistPath: defaultPersistPath,
		compress:    defaultCompress,
	}

	for _, opt := range opts {
		opt(s)
	}

	if s.embedder == nil {
		return nil, errors.New("embedder is required")
	}

	return s, nil
}
